import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# generate image.txt
def generate_img_txt(root):
    dir_list = os.listdir(root)
    f = open('images.txt','w')
    for diretory in dir_list:
        for img in os.listdir(os.path.join(root,diretory)):
            if img.endswith('RGB.jpg'):
                img_path = os.path.join(root,diretory,img)
                f.write(img_path+'\n')
    f.close()


# clustering to generate labels
def feature2label(features, cluster_num, dim_reduction='pca'):
    if dim_reduction == 'tsne':
        tsne = TSNE(n_components=2)
        X = tsne.fit_transform(features)
    else:
        pca = PCA(n_components=2)
        X = pca.fit_transform(features)
    label = KMeans(cluster_num).fit_predict(features)
    return X,label


# pretrain model
def pretrain(dataloader,device):
    pre_model = models.resnet18(pretrained=True).to(device)
    result = []
    for img in dataloader:
        pre_model.fc = nn.ReLU()
        pre_model.eval()
        with torch.no_grad():
            img = img.to(device)
            feature = pre_model(img).data.cpu().numpy().squeeze()
            result.append(feature)
    return result


def loss_plot(iteration,loss,round):
    plt.figure()
    plt.plot(iteration,loss)
    plt.savefig('images/Loss_Round_'+str(round+1) +'.jpg')
    # plt.show()


def cluster_plot(X,labels,round):
    colors = ['#0000FF','#1E90FF','#00BFFF','#87CEEB']
    plt.figure()
    for label in np.unique(labels):
        X_labeled = X[labels==label]
        plt.scatter(X_labeled[:,0],X_labeled[:,1],c=colors[label])
    plt.savefig('images/Kmeans_Round_'+str(round+1)+'.jpg')
    # plt.show()