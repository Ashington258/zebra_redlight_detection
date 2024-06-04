import cv2
import numpy as np

def on_trackbar_change(val):
    pass

def canny_edge_detection(image_path):
    # 读取图像
    original_image = cv2.imread(image_path)
    cv2.imshow('Original Image', original_image)
    
    # 创建Canny参数滑动条
    cv2.namedWindow('Canny Edges')
    cv2.createTrackbar('Threshold 1', 'Canny Edges', 100, 500, on_trackbar_change)
    cv2.createTrackbar('Threshold 2', 'Canny Edges', 200, 500, on_trackbar_change)
    
    while True:
        # 获取滑动条的值
        threshold1 = cv2.getTrackbarPos('Threshold 1', 'Canny Edges')
        threshold2 = cv2.getTrackbarPos('Threshold 2', 'Canny Edges')
        
        # 应用Canny边缘检测
        edges = cv2.Canny(original_image, threshold1, threshold2)
        
        # 显示Canny边缘检测后的图像
        cv2.imshow('Canny Edges', edges)
        
        # 按下 'Esc' 键退出循环
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# 调用函数进行测试
image_path = 'image\\2.png'
canny_edge_detection(image_path)
