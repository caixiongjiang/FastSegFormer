from ultralytics import YOLO


if __name__ == '__main__':
    # 载入一个模型
    model = YOLO('runs/segment/train/weights/best.pt')  # 载入自定义模型
    # model = YOLO('runs/segment/train2/weights/best.pt')  # 载入自定义模型

    # 验证模型
    metrics = model.val()  # 不需要参数，数据集和设置被记住了
    metrics.box.map    # map50-95(B)
    metrics.box.map50  # map50(B)
    metrics.box.map75  # map75(B)
    metrics.box.maps   # 各类别map50-95(B)列表
    metrics.seg.map    # map50-95(M)
    metrics.seg.map50  # map50(M)
    metrics.seg.map75  # map75(M)
    metrics.seg.maps   # 各类别map50-95(M)列表