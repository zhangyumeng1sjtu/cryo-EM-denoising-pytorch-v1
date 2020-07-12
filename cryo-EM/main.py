import argparse
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datetime import datetime
from torch import optim
from torch.utils.data import DataLoader
from dataset import Eval_Dataset, SiameseDataset
from models import SiameseNetwork, ContrastiveLoss
from utils import generate_img_txt, feature2label, pretrain, loss_plot, cluster_plot


def pre_work(txt,cluster_num,device):
    init_dataset = Eval_Dataset(txt,transforms.ToTensor(),initial=True)
    init_dataloader = DataLoader(init_dataset)

    init_feature = pretrain(init_dataloader,device)
    init_labellist = feature2label(init_feature,cluster_num)
    return init_labellist[1]


def train(labellist,batch_size,train_number_epochs,learning_rate,round,device):

    train_dataset = SiameseDataset('images.txt',labellist,transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    # training
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr=learning_rate)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(train_number_epochs):
        total_loss = 0
        start_time = datetime.now()
        for i,data in enumerate(train_dataloader):
            img0,img1,label = data
            img0,img1,label = img0.to(device),img1.to(device),label.to(device)

            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            total_loss += loss_contrastive.item()
            optimizer.step()
            if i%20 == 0:
                iteration_number += 20
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        end_time = datetime.now()
        print("Epoch number: {} , Current loss: {:.4f}, Epoch Time: {}".format(epoch+1,total_loss/(i+1),end_time-start_time))

    torch.save(net.state_dict(),"parameters/Parm_Round_"+str(round+1)+".pth")
    return counter, loss_history


def eval(txt,cluster_num,round,device):
    eval_dataset = Eval_Dataset('images.txt',transforms.ToTensor())
    eval_dataloader = DataLoader(eval_dataset)

    net = SiameseNetwork().to(device)
    net.load_state_dict(torch.load("parameters/Parm_Round_"+str(round+1)+".pth"))
    features = []

    for img in eval_dataloader:
        with torch.no_grad():
            img = img.to(device)
            output = net.forward_once(img).data.cpu().numpy().squeeze()
            features.append(output)
    
    X ,new_labellist = feature2label(features,cluster_num)
    return X, new_labellist


def show_average_img(cluster_num,round,labellist):
    eval_dataset = Eval_Dataset('images.txt',transforms.ToTensor())
    eval_dataloader = DataLoader(eval_dataset)
    plt.figure(figsize=(24,6))
    for i in range(cluster_num):
        plt.subplot(1,cluster_num,i+1)
        imglist = []
        for j,img in enumerate(eval_dataloader):
            if labellist[j] == i:
                imglist.append(img.squeeze().numpy())
        plt.imshow(np.mean(imglist,axis=0),cmap='gray')
    plt.savefig('images/Average_Result_Round_'+str(round+1)+'.jpg')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cryo-EM Images Denoising v1")
    parser.add_argument('-c','--cluster_num', default=4)
    parser.add_argument('-g','--gpu', default='cuda:0')
    parser.add_argument('-l','--learning_rate', default=5e-4)
    parser.add_argument('-e','--epoch_num', default=20)
    parser.add_argument('-b','--batch_size', default=128)
    parser.add_argument('-r','--round_num', default=5)
    parser.add_argument('-d','--dataset_dir', default='dataset')
    args = parser.parse_args()

    DEVICE = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epoch_num
    ROUND_NUM = args.round_num
    LR = args.learning_rate
    CLUSTER_NUM = args.cluster_num
    DATASET_DIR = args.dataset_dir

    print("----------------------Preparing-----------------------")
    generate_img_txt(root=DATASET_DIR)
    init_labellist = pre_work('images.txt',CLUSTER_NUM,DEVICE)
    show_average_img(CLUSTER_NUM,-1,init_labellist)
    for Round in range(ROUND_NUM):
        print("Round: %d" % (Round+1))
        if Round==0:
            labellist = init_labellist
        else:
            labellist = new_labellist
        print("----------------------Training-----------------------")
        counter, loss_history = train(labellist,BATCH_SIZE,NUM_EPOCHS,LR,Round,DEVICE)
        loss_plot(counter, loss_history, Round)
        print("----------------------Clustering-----------------------")
        X, new_labellist = eval('images.txt',CLUSTER_NUM,Round,DEVICE)
        cluster_plot(X, new_labellist, Round)
        show_average_img(CLUSTER_NUM,Round,new_labellist)