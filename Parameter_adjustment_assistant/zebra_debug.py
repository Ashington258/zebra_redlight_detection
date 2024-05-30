import cv2
import numpy as np

def on_trackbar_change(val):
    pass

def ensure_odd(val):
    return val if val % 2 == 1 else val + 1

def traffic_light_detection(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    original_image = image.copy()
    
    # 创建窗口和滑动条
    steps = ['1. Original Image', '2. Blurred Image', '3. Gaussian Blurred Image', '4. Red Mask', 
             '5. Canny Edges', '6. Morphologically Closed Edges']
    
    for step in steps:
        cv2.namedWindow(step)

    cv2.createTrackbar('Blur Kernel', '2. Blurred Image', 5, 30, on_trackbar_change)
    cv2.createTrackbar('Gaussian Kernel', '3. Gaussian Blurred Image', 5, 30, on_trackbar_change)
    cv2.createTrackbar('Low H', '4. Red Mask', 0, 179, on_trackbar_change)
    cv2.createTrackbar('High H', '4. Red Mask', 10, 179, on_trackbar_change)
    cv2.createTrackbar('Low S', '4. Red Mask', 100, 255, on_trackbar_change)
    cv2.createTrackbar('High S', '4. Red Mask', 255, 255, on_trackbar_change)
    cv2.createTrackbar('Low V', '4. Red Mask', 100, 255, on_trackbar_change)
    cv2.createTrackbar('High V', '4. Red Mask', 255, 255, on_trackbar_change)
    cv2.createTrackbar('Canny Threshold 1', '5. Canny Edges', 50, 200, on_trackbar_change)
    cv2.createTrackbar('Canny Threshold 2', '5. Canny Edges', 150, 300, on_trackbar_change)
    cv2.createTrackbar('Morph Kernel', '6. Morphologically Closed Edges', 5, 20, on_trackbar_change)
    cv2.createTrackbar('Min Area', '6. Morphologically Closed Edges', 500, 5000, on_trackbar_change)
    cv2.createTrackbar('Max Area', '6. Morphologically Closed Edges', 5000, 50000, on_trackbar_change)

    while True:
        # 获取滑动条的值
        blur_kernel_size = ensure_odd(cv2.getTrackbarPos('Blur Kernel', '2. Blurred Image'))
        gaussian_kernel_size = ensure_odd(cv2.getTrackbarPos('Gaussian Kernel', '3. Gaussian Blurred Image'))
        low_h = cv2.getTrackbarPos('Low H', '4. Red Mask')
        high_h = cv2.getTrackbarPos('High H', '4. Red Mask')
        low_s = cv2.getTrackbarPos('Low S', '4. Red Mask')
        high_s = cv2.getTrackbarPos('High S', '4. Red Mask')
        low_v = cv2.getTrackbarPos('Low V', '4. Red Mask')
        high_v = cv2.getTrackbarPos('High V', '4. Red Mask')
        canny_threshold1 = cv2.getTrackbarPos('Canny Threshold 1', '5. Canny Edges')
        canny_threshold2 = cv2.getTrackbarPos('Canny Threshold 2', '5. Canny Edges')
        morph_kernel_size = cv2.getTrackbarPos('Morph Kernel', '6. Morphologically Closed Edges')
        min_area = cv2.getTrackbarPos('Min Area', '6. Morphologically Closed Edges')
        max_area = cv2.getTrackbarPos('Max Area', '6. Morphologically Closed Edges')

        # 重置原始图像
        image = original_image.copy()

        # 平滑滤波
        blurred = cv2.medianBlur(image, blur_kernel_size)
        cv2.imshow('2. Blurred Image', blurred)
        
        # 高斯模糊
        gaussian_blurred = cv2.GaussianBlur(blurred, (gaussian_kernel_size, gaussian_kernel_size), 0)
        cv2.imshow('3. Gaussian Blurred Image', gaussian_blurred)

        # 转换为 HSV 色彩空间
        hsv = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2HSV)
        
        # 设置 HSV 阈值提取红色，并创建 mask 掩盖非红色区域
        lower_red1 = np.array([low_h, low_s, low_v])
        upper_red1 = np.array([high_h, high_s, high_v])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, low_s, low_v])
        upper_red2 = np.array([179, high_s, high_v])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        mask = mask1 + mask2
        cv2.imshow('4. Red Mask', mask)
        
        # canny 边缘检测
        edges = cv2.Canny(mask, canny_threshold1, canny_threshold2)
        cv2.imshow('5. Canny Edges', edges)
        
        # 形态学操作，侵蚀和膨胀让红灯轮廓闭合
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('6. Morphologically Closed Edges', closed_edges)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # 近似多边形
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                (x, y), radius = cv2.minEnclosingCircle(approx)
                center = (int(x), int(y))
                radius = int(radius)
                # 筛选圆形轮廓
                if len(approx) > 8 and radius > 5:
                    cv2.circle(image, center, radius, (0, 255, 0), 2)
                    detected = True

        cv2.imshow('1. Original Image', image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
            break

    cv2.destroyAllWindows()
    return detected

# 调用函数进行测试
image_path = 'image/2.png'
detected = traffic_light_detection(image_path)
print("Red Light Detected: ", detected)
