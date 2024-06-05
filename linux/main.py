#!/usr/bin/env python
#coding=utf-8
'''
Author: Ashington Ashington258@proton.me
Date: 2024-06-05 10:04:32
LastEditors: Ashington Ashington258@proton.me
LastEditTime: 2024-06-05 10:12:21
FilePath: /zebra_redlight_detection/linux/main.py
Description: 请填写简介
联系方式:921488837@qq.com
Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import cv2
import numpy as np
import multiprocessing


def ensure_odd(val):
    return val if val % 2 == 1 else val + 1

def zebra_crossing_detection(frame):
    # 设置默认参数值
    crop_percent = 0.3
    median_blur_val = 15
    gaussian_blur_val = 0
    canny_threshold1 = 50
    canny_threshold2 = 134
    morph_kernel_size = 1
    area_threshold = 3000
    hough_threshold = 16
    line_threshold = 10

    # 处理图像
    height = frame.shape[0]
    cropped_image = frame[int(crop_percent * height):, :]
    
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # 应用中值滤波
    if median_blur_val > 1:
        blurred = cv2.medianBlur(gray, median_blur_val)
    else:
        blurred = gray
    
    # 应用高斯模糊
    if gaussian_blur_val > 1:
        gaussian_blurred = cv2.GaussianBlur(blurred, (gaussian_blur_val, gaussian_blur_val), 0)
    else:
        gaussian_blurred = blurred
    
    edges = cv2.Canny(gaussian_blurred, canny_threshold1, canny_threshold2)
    
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed_edges)
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    
    roi = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_edges = cv2.Canny(roi_gray, canny_threshold1, canny_threshold2)
    
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=hough_threshold, minLineLength=50, maxLineGap=10)
    
    line_image = np.copy(cropped_image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    detected = lines is not None and len(lines) > line_threshold

    cv2.imshow('Detected Lines', line_image)
    
    if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
        return False

    return detected

def ensure_odd(val):
    return val if val % 2 == 1 else val + 1

def red_light_detection(frame):
    # 设置默认参数值
    smooth_kernel_size = 21
    gaussian_blur_val = 15
    lower_hue = 0
    upper_hue = 10
    lower_saturation = 100
    upper_saturation = 255
    lower_value = 100
    upper_value = 255
    morph_kernel_size = 5
    min_area = 400
    max_area = 3000

    # 平滑滤波
    smooth = cv2.medianBlur(frame, smooth_kernel_size)
    
    # 高斯模糊
    gaussian_blurred = cv2.GaussianBlur(smooth, (gaussian_blur_val, gaussian_blur_val), 0)
    
    # 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2HSV)
    
    # 提取红色区域
    lower_red = np.array([lower_hue, lower_saturation, lower_value])
    upper_red = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # 形态学操作，侵蚀和膨胀让红灯轮廓闭合
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = False
    output_image = frame.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circle_area = np.pi * (radius ** 2)
            
            if min_area < circle_area < max_area:
                cv2.circle(output_image, center, radius, (0, 255, 0), 2)
                detected = True
    
    cv2.imshow('Red Light Detection', output_image)
    
    if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
        return False

    return detected



def capture_frames(frame_queue):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()

def run_zebra_detection(frame_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            zebra_crossing_detection(frame)

def run_red_light_detection(frame_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            red_light_detection(frame)

if __name__ == "__main__":
    frame_queue = multiprocessing.Queue(maxsize=10)

    p1 = multiprocessing.Process(target=capture_frames, args=(frame_queue,))
    p2 = multiprocessing.Process(target=run_zebra_detection, args=(frame_queue,))
    p3 = multiprocessing.Process(target=run_red_light_detection, args=(frame_queue,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()


