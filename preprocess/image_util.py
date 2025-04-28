import cv2
import numpy as np

def sobel_gradient(image):
    """使用Sobel算子计算梯度幅值的平均值作为清晰度度量"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(sobelx**2 + sobely**2))

def select_image_with_low_blur(image_path_list):
    fm_list = []
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = sobel_gradient(gray)
        fm_list.append(fm)
    
    fms = -1
    highest_fm_idx = -1
    for idx, fm in enumerate(fm_list):
        if fm > fms:
            fms = fm
            highest_fm_idx = idx
    return image_path_list[highest_fm_idx]