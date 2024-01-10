# Embracing Uncertainty: The Impact of Fine-tuning Paradigms on Unknown Plant Diseases Recognition

* This code is an implementation of our manuscript: Embracing Uncertainty: The Impact of Fine-tuning Paradigms on Unknown Plant Diseases Recognition.

* Authors: Jiuqing Dong, et al.
* We will release the complete code and complete this documentation after the article is published.
* please refer more results in '[Result.xlsx](https://github.com/JiuqingDong/PDOOD/blob/main/Result.xlsx)'.

## Installation

Please check `env_setup.sh`.


## Pre-trained model preparation

Download and place the pre-trained Transformer-based backbones to './models/'. In our study, we use the [ViT-Base pre-trained](https://drive.google.com/file/d/11KuAkntDTPPcq4h4JwSjbDebNgVkgceA/view?usp=drive_link) on Imagenet-21k.

## Dataset Prepairation

  Cotton disease dataset: [https://www.kaggle.com/datasets/dhamur/cotton-plant-disease](https://www.kaggle.com/datasets/dhamur/cotton-plant-disease)
  
  Mango disease dataset: [https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)
  
  Strawberry disease dataset: [https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)
  
  Tomato disease dataset and Plant village dataset: [https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw)

please split the dataset by using our code.

## Training and Test
  We have provided the training scripts for `VPT, FFT, VAT, and LPT` in `train_train0.sh`, `train_train1.sh`, `train_train2.sh`, and `train_train3.sh`, respectively.
  We use VPT as an example. Run `CUDA_VISIBLE_DEVICES=0 sh train_gpu0.sh`.
  ### Key configs:
```
python train.py \                                       # For Stage 1
      --train-type "prompt"  \                          # VPT method
      --config-file configs/prompt/plantvillage.yaml  \   # the configuration of Imagenet-1k dataset
      DATA.PERCENTAGE '0.1' \                           # For the plant village dataset, 0.1, 0.2, 0.3, 0.4 correspond to 2, 4, 8, and 16 shots, respectively. While, for other datasets, it denotes experiments ID.
      DATA.FEATURE "sup_vitb16_imagenet21k" \           # specify which representation to use
      MODEL.TYPE "vit" \                                # the general backbone type, e.g., "vit" or "swin"
      DATA.BATCH_SIZE "128" \
      SOLVER.BASE_LR "0.1" \                            # learning rate for the experiment
      SOLVER.WEIGHT_DECAY "0.0" \                       # weight decay value for the experiment
      MODEL.PROMPT.DEEP "True" \                        # deep or shallow prompt
      MODEL.PROMPT.DROPOUT "0.1" \
      MODEL.PROMPT.NUM_TOKENS "10" \                    # prompt length
      MODEL.MODEL_ROOT "models/" \                      # folder with pre-trained model checkpoints
      OUTPUT_DIR "./FT_ood" \                         # output dir of the final model and logs
      SOLVER.TOTAL_EPOCH '100' \                    
      # SOLVER.TOTAL_EPOCH '0' \                        # for test phase
      # MODEL.WEIGHT_PATH "./FT_ood/plantvillage/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_Imagenet1k_best_model.pth" \

```

## Train

  sh train.sh



## We will release the complete code and complete this documentation after our manuscript is accepted.

## How to use this code for a customer dataset?






