# cyro-EM-denoising-pytorch-v1

## Principle
### Backgroud Info
Cryo-electron microscopy has gradually become an important technology in the field of structural biology. With the development and improvement of hardware and software, more and more molecular biological structures close to atomic resolution have been resolved. In order to obtain an accurate and reliable three-dimensional structure, it is a very important and critical step to perform cluster analysis on the projection images of cryo-electron microscopy.
### Siamese Network
1. Random choose two images from the dataset, in one cluster or not.

2. Send two images into the same networks (feature extractor), and obtain two latent vectors.

3. Use Contrastive Loss Function to evaluate the dissimilarity between two images.

$L = \frac{1}{2N}\sum_{n=1}^N(1-y)d^2+(y)\max(\text{margin}-d,0)^2$

## Usage
