### YOLOv8-Seg

[中文](README_CH.md)

#### catalogs

* `yolov8n-seg-train`: yolov8n-seg model training log.
* `yolov8s-seg-train`: yolov8s-seg model training log.
* `yolov8n-seg-val`: yolov8n-seg modeling test results.
* `yolov8s-seg-val`: yolov8s-seg modeling test results.
* `yolov8n-seg-predict`: Partial image prediction results for the yolov8n-seg model.
* `yolov8s-seg-predict`: Partial image prediction results for the yolov8s-seg model.
* `labels`: Part of the image label mask, which corresponds one-to-one with the one in the prediction folder.
* `mask2txt.py`: Transformation script for semantic segmentation tags into instance segmentation tags.
* `train.py`: Train scripts.
* `test.py`: Test scripts.
* `predict.py`: Predict scripts.

**Although the yolov8-seg family of models has an extremely fast inference speed, 
it is very unstable in the distinction between wind scarring and ulcer, 
and the example segmentation method is difficult to perform the task of detailed segmentation of this defect dataset.**


