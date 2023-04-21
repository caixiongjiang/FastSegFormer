## FastSegFormer: A knowledge distillation-based method for real-time semantic segmentation of surface defects in navel oranges

This is the official repository for our work: FastSegFormer([PDF]())

### Highlights

![]()

* ds 
* ds 
* ds

### Demos

A demo of the segmentation performance of our proposed FastSegFormer:Original image(left) and prediction of FastSegFormer-E(middle) and
FastSegFormer-P(right).




### Overview

* An overview of the architecture of our proposed FastSegFormer-P. The architecture of FastSegFormer-E is derived from FastSegFormer-P
replacing the backbone network EfficientFormerV2-S0.
![](Images/model.png)

* An overview of the proposed multi-resolution knowledge distillation.(To solve the problem that the size and number of channels of the teacher network and student
network feature maps are different:the teacher network's feature maps are down-sampled by bilinear interpolation, and the student network's feature maps
 are convolved point-by-point to increase the number of channels)
![](Images/Knowledge%20Distillation.png)

### Models

* Pretrained backbone network:

|   Dataset    |    Input size    | PoolFormer-S12  | EfficientFormerV2-S0  |
|:------------:|:----------------:|:---------------:|:---------------------:|
| ImageNet-1K  | $224\times 224$  |  [download]()   |     [download]()      |

* Teacher network:

|      Model       |   Input size    | mIoU  |  mPA  | params | GFLOPs |     ckpt     |
|:----------------:|:---------------:|:-----:|:-----:|:------:|:------:|:------------:|
| swin-T-Att-UNet  | $512\times 512$ | 90.53 | 94.65 | 49.21M | 77.80  | [download]() |

* FastSegFormer after fine-tuning and knowledge distillation:

| Model           |    Input size    | mIoU  |  mPA   | params | GFLOPs | RTX3060(FPS) | RTX3050Ti(FPS) |     ckpt     |
|:----------------|:----------------:|:-----:|:------:|:------:|:------:|:------------:|:--------------:|:------------:|
| FastSegFormer-E | $224\times 224$  | 88.78 | 93.33  | 5.01M  |  0.80  |      61      |       -        | [download]() |
| FastSegFormer-P | $224\times 224$  | 89.33 | 93.33  | 14.87M |  2.70  |     104      |       -        | [download]() |

