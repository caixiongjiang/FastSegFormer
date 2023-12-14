import cv2
import numpy as np
import os
from PIL import Image
import shutil



def make_path(file_path):
    os.makedirs(file_path, exist_ok=True)  

def tongji(train_caitu_dir, image_counts):
    # 统计图片信息
    cnt = 0
    for file in os.listdir(train_caitu_dir):
        if file.endswith(".png"):
            if cnt > 30:
                break
            png_path = os.path.join(train_caitu_dir, file) 
            image = Image.open(png_path)
            pixels = image.convert("RGB").getdata()
            
            for pixel in pixels:
                if pixel not in image_counts:
                    image_counts[pixel] = 1
                else:
                    image_counts[pixel] += 1
            cnt += 1         
    # 打印像素值统计结果
    for pixel, count in image_counts.items():
        print(f"Pixel: {pixel}, Count: {count}")
    
"""
Pixels:
(128, 0, 0), sunburn 
(128, 128, 0), wind scarring
(0, 128, 0), ulcer   
"""


def analyse_mask_save_txt(png_path, txt_path, colors):
    # 读取彩色掩码图像
    mask_image =  cv2.imread(png_path, cv2.IMREAD_COLOR)
    # 读取宽高
    img_width, img_height = mask_image.shape[:2]
    
    with open(txt_path, 'w') as file:
        for index, color in enumerate(colors):
            # print(color)
            mask = cv2.inRange(mask_image, np.array(color), np.array(color))
            counters, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
            for counter in counters:
                file.write(f"{index}")
                for point in counter.squeeze():
                    x, y = point.astype(np.float32) / [img_width, img_height]
                    file.write(f" {x:.3f} {y:.3f}")
                file.write("\n")
            

def copy_file(origin_file, target_file):
    try:
        shutil.copy(origin_file, target_file)
        print(f"{origin_file} --> {target_file}")
    except IOError as e:
        print(f"can't copy image. {e}")
    except:
        print(f"An error occurred and the picture was not copied")
    
    
  



if __name__ == '__main__':
    
    # RGB->BGR(CV2读取)
    SUNBURN = [0, 0, 128]
    ULCER = [0, 128, 0]
    WIND_SCARRING = [0, 128, 128]
    COLORS = [SUNBURN, ULCER, WIND_SCARRING]
    
    tongji(r"Orange-Navel-4.5k\caitu\train", {})
    root_dir = ".\Orange-Navel-4.5k"    
    img_root_dir = ".\Orange-Navel-4.5k\images"
    img_train_dir = os.path.join(img_root_dir, "train")
    img_val_dir = os.path.join(img_root_dir, "val")
    img_test_dir = os.path.join(img_root_dir, "test")
    png_root_dir = ".\Orange-Navel-4.5k\caitu"
    png_train_dir = os.path.join(png_root_dir, "train")
    png_val_dir = os.path.join(png_root_dir, "val")
    png_test_dir = os.path.join(png_root_dir, "test")
    label_root_dir = ".\Orange-Navel-4.5k\labels"
    label_train_dir = os.path.join(label_root_dir, "train")
    label_val_dir = os.path.join(label_root_dir, "val")
    label_test_dir = os.path.join(label_root_dir, "test")

    make_path(label_train_dir)
    make_path(label_val_dir)
    make_path(label_test_dir)

    # 训练数据标签txt文件生成
    for file in os.listdir(png_train_dir):
        if file.endswith(".png"):
            print(f"train images process:{file}")
            png_path = os.path.join(png_train_dir, file)
            label_path = os.path.join(label_train_dir, file.split(".")[0] + ".txt")
            analyse_mask_save_txt(png_path, label_path, COLORS)
            
    # 验证数据标签txt文件生成
    for file in os.listdir(png_val_dir):
        if file.endswith(".png"):
            print(f"val images process:{file}")
            png_path = os.path.join(png_val_dir, file)
            label_path = os.path.join(label_val_dir, file.split(".")[0] + ".txt")
            analyse_mask_save_txt(png_path, label_path, COLORS)

    # 测试数据标签txt文件生成
    for file in os.listdir(png_test_dir):
        if file.endswith(".png"):
            print(f"test images process:{file}")
            png_path = os.path.join(png_test_dir, file)
            label_path = os.path.join(label_test_dir, file.split(".")[0] + ".txt")
            analyse_mask_save_txt(png_path, label_path, COLORS)
            
    # # 按照yolo的数据集排列方式
    # train_dir = os.path.join(root_dir, "train")
    # val_dir = os.path.join(root_dir, "val")
    # test_dir = os.path.join(root_dir, "test")
    # new_train_img_dir = os.path.join(train_dir, "images")
    # new_val_img_dir = os.path.join(val_dir, "images")
    # new_test_img_dir = os.path.join(test_dir, "images")
    # new_train_label_dir = os.path.join(train_dir, "labels")
    # new_val_label_dir = os.path.join(val_dir, "labels")
    # new_test_label_dir = os.path.join(test_dir, "labels")
    # make_path(new_train_img_dir)
    # make_path(new_val_img_dir)
    # make_path(new_test_img_dir)
    # make_path(new_train_label_dir)
    # make_path(new_val_label_dir)
    # make_path(new_test_label_dir)    

    # # 复制文件到指定路径
    # for file in os.listdir(img_train_dir):
    #     origin_file = os.path.join(img_train_dir, file)
    #     target_file = os.path.join(new_train_img_dir, file)
    #     copy_file(origin_file, target_file)
    
    # for file in os.listdir(img_val_dir):
    #     origin_file = os.path.join(img_val_dir, file)
    #     target_file = os.path.join(new_val_img_dir, file)
    #     copy_file(origin_file, target_file)
        
    # for file in os.listdir(img_test_dir):
    #     origin_file = os.path.join(img_test_dir, file)
    #     target_file = os.path.join(new_test_img_dir, file)
    #     copy_file(origin_file, target_file)
        
    # for file in os.listdir(label_train_dir):
    #     origin_file = os.path.join(label_train_dir, file)
    #     target_file = os.path.join(new_train_label_dir, file)
    #     copy_file(origin_file, target_file)
        
    # for file in os.listdir(label_val_dir):
    #     origin_file = os.path.join(label_val_dir, file)
    #     target_file = os.path.join(new_val_label_dir, file)
    #     copy_file(origin_file, target_file)
        
    # for file in os.listdir(label_test_dir):
    #     origin_file = os.path.join(label_test_dir, file)
    #     target_file = os.path.join(new_test_label_dir, file)
    #     copy_file(origin_file, target_file)
        
        




