#!/usr/bin/env python
#coding=utf-8
'''
Author: Ashington Ashington258@proton.me
Date: 2024-06-05 10:22:39
LastEditors: Ashington Ashington258@proton.me
LastEditTime: 2024-06-05 10:22:39
FilePath: /zebra_redlight_detection/linux/camera_list.py
Description: 请填写简介
联系方式:921488837@qq.com
Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import cv2

def list_cameras():
    # 列出可用的摄像头
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def open_camera(index):
    # 打开指定索引的摄像头
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"无法打开索引为 {index} 的摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧（可能是摄像头断开）")
            break
        
        cv2.imshow(f'摄像头 {index}', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 列出所有可用的摄像头
    cameras = list_cameras()
    if not cameras:
        print("没有找到可用的摄像头")
    else:
        print("可用的摄像头索引: ", cameras)
        
        # 打开第一个可用的摄像头进行测试
        open_camera(cameras[0])
