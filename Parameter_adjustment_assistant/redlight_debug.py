import cv2
import numpy as np

def ensure_odd(val):
    return val if val % 2 == 1 else val + 1

def on_trackbar_change(val):
    pass

def traffic_light_detection(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 创建窗口和滑动条
    steps = ['1. Original Image', '2. HSV Image', '3. Smoothed Image', '4. Gaussian Blurred Image', '5. Red Mask', '6. Canny Edges', '7. Morphologically Closed Edges', '8. Filtered Contours', '9. Detected Red Lights']
    
    for step in steps:
        cv2.namedWindow(step)

    cv2.createTrackbar('Lower H', '2. HSV Image', 0, 255, on_trackbar_change)
    cv2.createTrackbar('Upper H', '2. HSV Image', 10, 255, on_trackbar_change)
    cv2.createTrackbar('Lower S', '2. HSV Image', 70, 255, on_trackbar_change)
    cv2.createTrackbar('Upper S', '2. HSV Image', 255, 255, on_trackbar_change)
    cv2.createTrackbar('Lower V', '2. HSV Image', 50, 255, on_trackbar_change)
    cv2.createTrackbar('Upper V', '2. HSV Image', 255, 255, on_trackbar_change)
    
    cv2.createTrackbar('Median Blur', '3. Smoothed Image', 5, 20, on_trackbar_change)
    cv2.createTrackbar('Gaussian Blur', '4. Gaussian Blurred Image', 5, 15, on_trackbar_change)
    cv2.createTrackbar('Canny Threshold 1', '6. Canny Edges', 100, 300, on_trackbar_change)
    cv2.createTrackbar('Canny Threshold 2', '6. Canny Edges', 200, 300, on_trackbar_change)
    cv2.createTrackbar('Morph Kernel', '7. Morphologically Closed Edges', 5, 20, on_trackbar_change)
    cv2.createTrackbar('Min Area', '8. Filtered Contours', 500, 5000, on_trackbar_change)
    cv2.createTrackbar('Max Area', '8. Filtered Contours', 3000, 10000, on_trackbar_change)
    cv2.createTrackbar('Circularity Min', '9. Detected Red Lights', 70, 100, on_trackbar_change)
    cv2.createTrackbar('Circularity Max', '9. Detected Red Lights', 120, 200, on_trackbar_change)

    while True:
        # 获取滑动条的值
        lower_h = cv2.getTrackbarPos('Lower H', '2. HSV Image')
        upper_h = cv2.getTrackbarPos('Upper H', '2. HSV Image')
        lower_s = cv2.getTrackbarPos('Lower S', '2. HSV Image')
        upper_s = cv2.getTrackbarPos('Upper S', '2. HSV Image')
        lower_v = cv2.getTrackbarPos('Lower V', '2. HSV Image')
        upper_v = cv2.getTrackbarPos('Upper V', '2. HSV Image')
        
        median_blur_val = ensure_odd(cv2.getTrackbarPos('Median Blur', '3. Smoothed Image'))
        gaussian_blur_val = ensure_odd(cv2.getTrackbarPos('Gaussian Blur', '4. Gaussian Blurred Image'))
        canny_threshold1 = cv2.getTrackbarPos('Canny Threshold 1', '6. Canny Edges')
        canny_threshold2 = cv2.getTrackbarPos('Canny Threshold 2', '6. Canny Edges')
        morph_kernel_size = cv2.getTrackbarPos('Morph Kernel', '7. Morphologically Closed Edges')
        min_area = cv2.getTrackbarPos('Min Area', '8. Filtered Contours')
        max_area = cv2.getTrackbarPos('Max Area', '8. Filtered Contours')
        circularity_min = cv2.getTrackbarPos('Circularity Min', '9. Detected Red Lights') / 100.0
        circularity_max = cv2.getTrackbarPos('Circularity Max', '9. Detected Red Lights') / 100.0
        
        # 1. 显示原始图像
        cv2.imshow('1. Original Image', image)
        
        # 2. BGR空间转HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imshow('2. HSV Image', hsv)

        # 3. 平滑滤波
        smoothed = cv2.medianBlur(hsv, median_blur_val)
        cv2.imshow('3. Smoothed Image', smoothed)

        # 4. 高斯模糊
        blurred = cv2.GaussianBlur(smoothed, (gaussian_blur_val, gaussian_blur_val), 0)
        cv2.imshow('4. Gaussian Blurred Image', blurred)

        # 5. 设置HSV阈值提取红色，并创建mask掩盖非红色区域
        lower_red1 = np.array([lower_h, lower_s, lower_v])
        upper_red1 = np.array([upper_h, upper_s, upper_v])
        lower_red2 = np.array([170, lower_s, lower_v])
        upper_red2 = np.array([180, upper_s, upper_v])

        mask1 = cv2.inRange(blurred, lower_red1, upper_red1)
        mask2 = cv2.inRange(blurred, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        cv2.imshow('5. Red Mask', mask)

        # 6. Canny边缘检测
        edges = cv2.Canny(mask, canny_threshold1, canny_threshold2)
        cv2.imshow('6. Canny Edges', edges)

        # 7. 形态学操作，侵蚀和膨胀让红灯轮廓闭合
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('7. Morphologically Closed Edges', closed)

        # 8. 设置阈值，过滤掉过小或过大得闭合区域
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

        # 创建一个空图像来绘制筛选后的轮廓
        filtered_contours_img = np.zeros_like(image)
        cv2.drawContours(filtered_contours_img, filtered_contours, -1, (0, 255, 0), 3)
        cv2.imshow('8. Filtered Contours', filtered_contours_img)

        # 9. 形态学筛选出圆形得图形，并输出检测到红灯，返回True
        detected = False
        for cnt in filtered_contours:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * cv2.contourArea(cnt) / (perimeter * perimeter)
            if circularity_min < circularity < circularity_max:
                detected = True
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)

        cv2.imshow('9. Detected Red Lights', image)

        if detected:
            print("Red light detected: True")
        else:
            print("Red light detected: False")

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

traffic_light_detection('image\\2.png')
