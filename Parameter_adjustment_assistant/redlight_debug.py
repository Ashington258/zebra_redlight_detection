import cv2
import numpy as np

def on_trackbar_change(val):
    pass

def ensure_odd(val):
    return val if val % 2 == 1 else val + 1

def red_light_detection(image_path):
    # 读取图像
    original_image = cv2.imread(image_path)
    
    # 创建窗口和滑动条
    steps = [
        '1. Original Image', '2. Smooth Image', '3. Gaussian Blurred Image', 
        '4. HSV Red Mask', '5. Morphological Transform', '6. Contours and Red Light'
    ]
    
    for step in steps:
        cv2.namedWindow(step)

    cv2.createTrackbar('Smooth Kernel', '2. Smooth Image', 20, 30, on_trackbar_change)
    cv2.createTrackbar('Gaussian Blur', '3. Gaussian Blurred Image', 10, 50, on_trackbar_change)
    cv2.createTrackbar('Lower Hue', '4. HSV Red Mask', 0, 180, on_trackbar_change)
    cv2.createTrackbar('Upper Hue', '4. HSV Red Mask', 10, 180, on_trackbar_change)
    cv2.createTrackbar('Lower Saturation', '4. HSV Red Mask', 100, 255, on_trackbar_change)
    cv2.createTrackbar('Upper Saturation', '4. HSV Red Mask', 255, 255, on_trackbar_change)
    cv2.createTrackbar('Lower Value', '4. HSV Red Mask', 100, 255, on_trackbar_change)
    cv2.createTrackbar('Upper Value', '4. HSV Red Mask', 255, 255, on_trackbar_change)
    cv2.createTrackbar('Erode/Dilate Kernel', '5. Morphological Transform', 3, 20, on_trackbar_change)
    cv2.createTrackbar('Min Area', '6. Contours and Red Light', 400, 5000, on_trackbar_change)
    cv2.createTrackbar('Max Area', '6. Contours and Red Light', 3000, 5000, on_trackbar_change)
    cv2.createTrackbar('Min Aspect Ratio', '6. Contours and Red Light', 75, 100, on_trackbar_change)
    cv2.createTrackbar('Max Aspect Ratio', '6. Contours and Red Light', 125, 150, on_trackbar_change)

    while True:
        # 复制原始图像，保证每次都从原始图像开始处理
        image = original_image.copy()

        # 获取滑动条的值
        smooth_kernel_size = ensure_odd(cv2.getTrackbarPos('Smooth Kernel', '2. Smooth Image'))
        gaussian_blur_val = ensure_odd(cv2.getTrackbarPos('Gaussian Blur', '3. Gaussian Blurred Image'))
        lower_hue = cv2.getTrackbarPos('Lower Hue', '4. HSV Red Mask')
        upper_hue = cv2.getTrackbarPos('Upper Hue', '4. HSV Red Mask')
        lower_saturation = cv2.getTrackbarPos('Lower Saturation', '4. HSV Red Mask')
        upper_saturation = cv2.getTrackbarPos('Upper Saturation', '4. HSV Red Mask')
        lower_value = cv2.getTrackbarPos('Lower Value', '4. HSV Red Mask')
        upper_value = cv2.getTrackbarPos('Upper Value', '4. HSV Red Mask')
        morph_kernel_size = ensure_odd(cv2.getTrackbarPos('Erode/Dilate Kernel', '5. Morphological Transform'))
        min_area = cv2.getTrackbarPos('Min Area', '6. Contours and Red Light')
        max_area = cv2.getTrackbarPos('Max Area', '6. Contours and Red Light')
        min_aspect_ratio = cv2.getTrackbarPos('Min Aspect Ratio', '6. Contours and Red Light') / 100.0
        max_aspect_ratio = cv2.getTrackbarPos('Max Aspect Ratio', '6. Contours and Red Light') / 100.0

        # 显示原始图像
        cv2.imshow('1. Original Image', original_image)

        # 平滑滤波
        smooth = cv2.medianBlur(image, smooth_kernel_size)
        cv2.imshow('2. Smooth Image', smooth)
        
        # 高斯模糊
        gaussian_blurred = cv2.GaussianBlur(smooth, (gaussian_blur_val, gaussian_blur_val), 0)
        cv2.imshow('3. Gaussian Blurred Image', gaussian_blurred)
        
        # 转换到 HSV 色彩空间
        hsv = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2HSV)
        
        # 提取红色区域
        lower_red = np.array([lower_hue, lower_saturation, lower_value])
        upper_red = np.array([upper_hue, upper_saturation, upper_value])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        cv2.imshow('4. HSV Red Mask', mask)
        
        # 形态学操作，侵蚀和膨胀让红灯轮廓闭合
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('5. Morphological Transform', morph)
        
        # 查找轮廓
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        output_image = original_image.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:  # 近似圆形
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    detected = True
        
        cv2.imshow('6. Contours and Red Light', output_image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
            break

    cv2.destroyAllWindows()
    return detected

# 调用函数进行测试
image_path = 'image\\RED_LIGHT.png'
detected = red_light_detection(image_path)
print("Red Light Detected: ", detected)
