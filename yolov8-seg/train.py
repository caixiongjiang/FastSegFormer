from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')
    # model = YOLO('yolov8s-seg.pt')

    results = model.train(data='./data.yaml', epochs=300, imgsz=224, batch=8, workers=2)


