import cv2
import numpy as np

def ensure_odd(val):
    return val if val % 2 == 1 else val + 1

def red_light_detection():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

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

    while True:
        # 从摄像头捕获一帧
        ret, original_image = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 平滑滤波
        smooth = cv2.medianBlur(original_image, smooth_kernel_size)
        
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
        output_image = original_image.copy()
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
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected

# 调用函数进行测试
detected = red_light_detection()
print("Red Light Detected: ", detected)
