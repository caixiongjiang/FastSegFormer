## FastSegFormer: A knowledge distillation-based method for real-time semantic segmentation of surface defects in navel oranges

This is the official repository for our work:FastSegFormer([PDF]())

### Highlights

---
![]()

* ds 
* ds 
* ds

### Demos

---
A demo of the segmentation performance of our proposed FastSegFormer:Original image(left) and prediction of FastSegFormer-E(middle) and
FastSegFormer-P(right).




### Models && Knowledge Distillation

---
* An overview of the architecture of our proposed FastSegFormer-P. The architecture of FastSegFormer-E is derived from FastSegFormer-P
replacing the backbone network EfficientFormerV2-S0.
![](Images/model.png)

* An overview of the proposed multi-resolution knowledge distillation.(To solve the problem that the size and number of channels of the teacher network and student
network feature maps are different:the teacher network's feature maps are down-sampled by bilinear interpolation, and the student network's feature maps
 are convolved point-by-point to increase the number of channels.)
![](Images/Knowledge%20Distillation.png)

