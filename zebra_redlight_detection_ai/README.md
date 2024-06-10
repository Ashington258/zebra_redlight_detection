<!--
 * @Author: Ashington ashington258@proton.me
 * @Date: 2024-06-09 14:22:22
 * @LastEditors: Ashington ashington258@proton.me
 * @LastEditTime: 2024-06-10 20:41:50
 * @FilePath: \zebra_redlight_detection\zebra_redlight_detection_ai\README.md
 * @Description: 请填写简介
 * 联系方式:921488837@qq.com
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
-->
# 说明

该工程基于ACL开发的神经网络用于检测斑马线和红绿灯

## 工程结构

📦zebra_redlight_detection_ai
 ┣ 📂roslibpy`该目录接入roslibpy，用于和ROS系统进行通信`
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜det_utils.cpython-39.pyc
 ┃ ┣ 📜data.yaml
 ┃ ┣ 📜det_utils.py
 ┃ ┣ 📜infer.launch
 ┃ ┣ 📜infer.py`程序入口`
 ┃ ┣ 📜test.py
 ┃ ┗ 📜yolov5s_bs1.om
 ┣ 📂src`该目录存放src源码，用于单独运行测试代码`
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜det_utils.cpython-39.pyc
 ┃ ┣ 📜data.yaml
 ┃ ┣ 📜det_utils.py
 ┃ ┣ 📜infer.launch
 ┃ ┣ 📜infer.py`程序入口`
 ┃ ┗ 📜yolov5s_bs1.om
 ┣ 📂subscriber`该目录为ROS系统下的订阅者`
 ┃ ┗ 📜subscriber.py
 ┣ 📂__pycache__
 ┃ ┗ 📜det_utils.cpython-39.pyc
 ┗ 📜README.md


单独测试时请运行:`python infer --model=yolov5s_bs1.om`
需要在ROS系统运行：
    1. 开启roscore，并且启动rosbridge`roslaunch rosbridge_server rosbridge_websocket.launch`
    2. 再启动roslib下python程序，注：此时必须有Ascend必要环境


## 模型


模型使用的数据集为：F:\15_Train_data\zebra_redlight\train_data_2024_6_8_10_49