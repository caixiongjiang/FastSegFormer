### YOLOv8-Seg

[English](README.md)

#### 目录

* `yolov8n-seg-train`: yolov8n-seg模型训练日志。
* `yolov8s-seg-train`: yolov8s-seg模型训练日志。
* `yolov8n-seg-val`: yolov8n-seg模型测试结果。
* `yolov8s-seg-val`: yolov8s-seg模型测试结果。
* `yolov8n-seg-predict`: yolov8n-seg模型部分图片预测结果。
* `yolov8s-seg-predict`: yolov8s-seg模型部分图片预测结果。
* `labels`: 部分图片标签掩码，与预测文件夹中的一一对应。
* `mask2txt.py`: 语义分割标签变为实例分割标签的转化脚本。
* `train.py`: 训练脚本。
* `test.py`: 测试脚本。
* `predict.py`: 预测图片脚本。

**尽管yolov8-seg系列模型拥有极快的推理速度，
但其在wind-scarring和ulcer的区分上很不稳定，
该实例分割方法难以胜任该缺陷数据集的细致化分割任务**