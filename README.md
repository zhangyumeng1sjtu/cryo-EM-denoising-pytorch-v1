# cyro-EM-denoising-pytorch-v1

## Principle
### Backgroud Info
Cryo-electron microscopy has gradually become an important technology in the field of structural biology. With the development and improvement of hardware and software, more and more molecular biological structures close to atomic resolution have been resolved. In order to obtain an accurate and reliable three-dimensional structure, it is a very important and critical step to perform cluster analysis on the projection images of cryo-electron microscopy.
### Siamese Network
1. Random choose two images from the dataset, in one cluster or not.

2. Send two images into the same networks (feature extractor), and obtain two latent vectors.

3. Use Contrastive Loss Function to evaluate the dissimilarity between two images.

$$L = \frac{1}{2N}\sum_{n=1}^N(1-y)d^2+(y)\max(\text{margin}-d,0)^2$$

y = 1 stands for the two images come from different clusters, 
y = 0 stands for the two images come from same clusters.

4. Kmeans used to assign new labels.

```python
# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,4,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4,8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8,8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*64*64,500),
            nn.ReLU(inplace=True),

            nn.Linear(500,100),
            nn.ReLU(inplace=True),

            nn.Linear(100,20)
        )

    def forward_once(self,x):
        output = self.cnn1(x)
        output = output.view(output.size()[0],-1)
        output = self.fc1(output)
        return output

    def forward(self,input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2  
```

## Usage
