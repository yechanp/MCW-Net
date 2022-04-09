# MCW-Net



# MCW-Net: Single Image Deraining with Multi-level Connections and Wide Regional Non-local Blocks

## Abstract

A recent line of convolutional neural network-based works has succeeded in capturing rain streaks. However, difficulties in detailed recovery still remain. In this paper, we present a multi-level connection and wide regional non-local block network (MCW-Net) to properly restore the original background textures in rainy images. Unlike existing encoder-decoder-based image deraining models that improve performance with additional branches, MCW-Net improves performance by maximizing information utilization without additional branches through the following two proposed methods. The first method is a multi-level connection that repeatedly connects multi-level features of the encoder network to the decoder network. Multi-level connection encourages the decoding process to use the feature information of all levels. In multi-level connection, channel-wise attention is considered to learn which level of features is important in the decoding process of the current level. The second method is a wide regional non-local block. As rain streaks primarily exhibit a vertical distribution, we divide the grid of the image into horizontally-wide patches and apply a non-local operation to each region to explore the rich rain-free background information. Experimental results on both synthetic and real-world rainy datasets demonstrate that the proposed model significantly outperforms existing state-of-the-art models. Furthermore, the results of the joint deraining and segmentation experiment prove that our model contributes effectively to other vision tasks.



# Experiments
## Requirements


## Dataset Preparation

## Usage

### Pretrained weights
Download the weights in https://drive.google.com/drive/folders/1_Dkr4CAPxWHbkOU7PIV3gbg9-oqotWRI?usp=sharing

### Training


