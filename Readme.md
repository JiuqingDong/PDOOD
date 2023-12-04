# Embracing Uncertainty- Efficient Tuning is a Key for Unknown Plant Disease Recognition

* This code is an implementation of our manuscript: Embracing Uncertainty- Efficient Tuning is a Key for Unknown Plant Disease Recognition.

* Authors: Jiuqing Dong, et al.
* We will release the complete code and complete this documentation after the article is published.
* please refer more results in '[Result.xlsx](https://github.com/JiuqingDong/PDOOD/blob/main/Result.xlsx)'.

## Installation
* Set up environment

* install dependecies

## Pre-trained model preparation

Download and place the pre-trained Transformer-based backbones to './models/'. In our study, we use the [ViT-Base pre-trained](https://drive.google.com/file/d/11KuAkntDTPPcq4h4JwSjbDebNgVkgceA/view?usp=drive_link) on Imagenet-21k.

## dataset prepair

  Cotton disease dataset: [https://www.kaggle.com/datasets/dhamur/cotton-plant-disease](https://www.kaggle.com/datasets/dhamur/cotton-plant-disease)
  
  Mango disease dataset: [https://www.kaggle.com/datasets/dhamur/cotton-plant-disease](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)
  
  Strawberry disease dataset: [https://www.kaggle.com/datasets/dhamur/cotton-plant-disease](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)
  
  Tomato disease dataset and Plant village dataset: [https://www.kaggle.com/datasets/dhamur/cotton-plant-disease](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)

please split the dataset by using our code.

## train

  sh train.sh

## test

  sh test.sh

## We will release the complete code and complete this documentation after our manuscript is accepted.

## How to use this code for a customer dataset?






