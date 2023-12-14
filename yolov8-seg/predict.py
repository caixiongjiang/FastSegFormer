from ultralytics import YOLO

if __name__ == '__main__':
    
    # 载入一个模型
    # model = YOLO('runs/segment/yolov8n-seg-train/weights/best.pt')  # 载入自定义模型
    model = YOLO('runs/segment/yolov8s-seg-train/weights/best.pt')  # 载入自定义模型

    # 使用模型进行预测
    results = model('Orange-Navel-4.5k/images/test/img31.jpg', save=True)  
    results = model('Orange-Navel-4.5k/images/test/img200.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img351.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img431.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img534.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img544.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img685.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img1110.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img1234.jpg', save=True)
    results = model('Orange-Navel-4.5k/images/test/img1448.jpg', save=True)