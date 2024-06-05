#!/usr/bin/env python
#coding=utf-8
'''
Author: Ashington Ashington258@proton.me
Date: 2024-06-05 10:15:23
LastEditors: Ashington Ashington258@proton.me
LastEditTime: 2024-06-05 10:16:04
FilePath: /zebra_redlight_detection/linux/camera_test.py
Description: 请填写简介
联系方式:921488837@qq.com
Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import cv2

# 打开默认摄像头（通常是设备的第一个摄像头）
cap = cv2.VideoCapture(1)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取一帧
    ret, frame = cap.read()
    
    # 检查是否成功读取帧
    if not ret:
        print("无法接收帧（可能是摄像头断开）")
        break
    
    # 显示帧
    cv2.imshow('摄像头测试', frame)
    
    # 按下 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
