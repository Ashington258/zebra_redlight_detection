# 说明

## 工程结构

📦zebra_crossing_detection
 ┣ 📂doc
 ┃ ┣ 📂工程结构
 ┃ ┃ ┗ 📜工程结构.md
 ┃ ┣ 📂斑马线流程
 ┃ ┃ ┣ 📜代码流程.ipynb
 ┃ ┃ ┗ 📜整体流程.md
 ┃ ┣ 📂红绿灯流程
 ┃ ┃ ┗ 📜流程.md
 ┃ ┗ 📂论文资料
 ┃ ┃ ┣ 📂OpenCV
 ┃ ┃ ┃ ┣ 📜斑马线智能分割及行人位置自动判别方法研究_李平原.pdf
 ┃ ┃ ┃ ┗ 📜车载激光点云斑马线提取方法研究_王耀.pdf
 ┃ ┃ ┗ 📂人工智能
 ┃ ┃ ┃ ┣ 📜基于改进SegNet模型的斑马线检测方法研究_付阳阳.pdf
 ┃ ┃ ┃ ┗ 📜基于深度学习的道路斑马线与行人运动检测算法研究与应用_付阳阳.pdf
 ┣ 📂image
 ┃ ┣ 📜1.png
 ┃ ┣ 📜1_gray.png
 ┃ ┣ 📜2.png
 ┃ ┣ 📜3.png
 ┃ ┣ 📜fe9ee2fae4d6cc95f3bce01334d1016.jpg
 ┃ ┗ 📜RED_LIGHT.png
 ┣ 📂Parameter_adjustment_assistant
 ┃ ┣ 📂traffic_detection
 ┃ ┃ ┣ 📜redlight_debug.py
 ┃ ┃ ┗ 📜red_live_debug.py
 ┃ ┗ 📂zebra_detection
 ┃ ┃ ┣ 📜zebra_debug.py
 ┃ ┃ ┗ 📜zebra_live_debug.py
 ┣ 📂test
 ┃ ┣ 📜camera_test.py
 ┃ ┣ 📜canny_debug.py
 ┃ ┣ 📜Grayscale_threshold_debug.py
 ┃ ┣ 📜HSV_debug.py
 ┃ ┣ 📜red_light_detection.py
 ┃ ┗ 📜zebra_crossing_detection.py
 ┣ 📂traffic_detection
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜red_light_detection.cpython-38.pyc
 ┃ ┗ 📜red_light_detection.py
 ┣ 📂zebra_detection
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜zebra_crossing_detection.cpython-38.pyc
 ┃ ┗ 📜zebra_crossing_detection.py
 ┣ 📜.gitignore
 ┣ 📜main.py
 ┗ 📜README.md

## 调用说明

📦zebra_crossing_detection
 ┣ 📂doc **(存放一些文档和参考资料)**
 ┣ 📂Parameter_adjustment_assistant **（该文件加用于调整参数，live为动态调整，其余为静态调整）**
 ┣ 📂test **（存放一些测试文件，例如测试摄像头和部分算法）**
 ┣ 📂traffic_detection **（存放红绿灯检测代码）**
 ┣ 📂zebra_detection **（存放斑马线检测代码）**
 ┣ 📜main.py **（函数入口，使用多线程调用）**
