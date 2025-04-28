import cv2
import numpy as np

def variance_of_laplacian(image):
    # 计算拉普拉斯算子
    return cv2.Laplacian(image, cv2.CV_64F).var()

def select_image_with_low_blur(image_path_list):
    fm_list = []
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        fm_list.append(fm)
    
    fms = -1
    highest_fm_idx = -1
    for idx, fm in enumerate(fm_list):
        if fm > fms:
            fms = fm
            highest_fm_idx = idx
    return image_path_list[highest_fm_idx]