import cv2
import numpy as np

def on_trackbar_change(val):
    pass

def ensure_odd(val):
    return val if val % 2 == 1 else val + 1

def zebra_crossing_detection():
    cap = cv2.VideoCapture(2)  # 使用摄像头
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    # 创建窗口和滑动条
    steps = ['1. Cropped Image', '2. Gray Image', '3. Median Blurred Image', '4. Gaussian Blurred Image', 
             '5. Canny Edges', '6. Morphologically Closed Edges', '7. Mask with Contours', 
             '8. ROI Image', '9. ROI Canny Edges', '10. Detected Lines']
    
    for step in steps:
        cv2.namedWindow(step)

    cv2.createTrackbar('Crop %', '1. Cropped Image', 30, 100, on_trackbar_change)
    cv2.createTrackbar('Median Blur', '3. Median Blurred Image', 15, 20, on_trackbar_change)
    cv2.createTrackbar('Gaussian Blur', '4. Gaussian Blurred Image', 0, 15, on_trackbar_change)
    cv2.createTrackbar('Canny Threshold 1', '5. Canny Edges', 50, 200, on_trackbar_change)
    cv2.createTrackbar('Canny Threshold 2', '5. Canny Edges', 134, 300, on_trackbar_change)
    cv2.createTrackbar('Morph Kernel', '6. Morphologically Closed Edges', 1, 20, on_trackbar_change)
    cv2.createTrackbar('Area Threshold', '6. Morphologically Closed Edges', 3000, 5000, on_trackbar_change)
    cv2.createTrackbar('Hough Threshold', '10. Detected Lines', 16, 200, on_trackbar_change)
    cv2.createTrackbar('Line Threshold', '10. Detected Lines', 10, 50, on_trackbar_change)

    while True:
        ret, frame = cap.read()  # 从摄像头读取帧
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # 获取滑动条的值
        crop_percent = cv2.getTrackbarPos('Crop %', '1. Cropped Image') / 100.0
        median_blur_val = ensure_odd(cv2.getTrackbarPos('Median Blur', '3. Median Blurred Image'))
        gaussian_blur_val = ensure_odd(cv2.getTrackbarPos('Gaussian Blur', '4. Gaussian Blurred Image'))
        canny_threshold1 = cv2.getTrackbarPos('Canny Threshold 1', '5. Canny Edges')
        canny_threshold2 = cv2.getTrackbarPos('Canny Threshold 2', '5. Canny Edges')
        morph_kernel_size = cv2.getTrackbarPos('Morph Kernel', '6. Morphologically Closed Edges')
        area_threshold = cv2.getTrackbarPos('Area Threshold', '6. Morphologically Closed Edges')
        hough_threshold = cv2.getTrackbarPos('Hough Threshold', '10. Detected Lines')
        line_threshold = cv2.getTrackbarPos('Line Threshold', '10. Detected Lines')

        # 处理图像
        height = frame.shape[0]
        cropped_image = frame[int(crop_percent * height):, :]
        cv2.imshow('1. Cropped Image', cropped_image)
        
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('2. Gray Image', gray)
        
        # 应用中值滤波
        if median_blur_val > 1:
            blurred = cv2.medianBlur(gray, median_blur_val)
        else:
            blurred = gray
        cv2.imshow('3. Median Blurred Image', blurred)
        
        # 应用高斯模糊
        if gaussian_blur_val > 1:
            gaussian_blurred = cv2.GaussianBlur(blurred, (gaussian_blur_val, gaussian_blur_val), 0)
        else:
            gaussian_blurred = blurred
        cv2.imshow('4. Gaussian Blurred Image', gaussian_blurred)
        
        edges = cv2.Canny(gaussian_blurred, canny_threshold1, canny_threshold2)
        cv2.imshow('5. Canny Edges', edges)
        
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('6. Morphologically Closed Edges', closed_edges)
        
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(closed_edges)
        for contour in contours:
            if cv2.contourArea(contour) > area_threshold:
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        cv2.imshow('7. Mask with Contours', mask)
        
        roi = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        cv2.imshow('8. ROI Image', roi)
        
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, canny_threshold1, canny_threshold2)
        cv2.imshow('9. ROI Canny Edges', roi_edges)
        
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=hough_threshold, minLineLength=50, maxLineGap=10)
        
        line_image = np.copy(cropped_image)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('10. Detected Lines', line_image)
        
        detected = lines is not None and len(lines) > line_threshold
        
        if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected

# 调用函数进行测试
detected = zebra_crossing_detection()
print("Zebra Crossing Detected: ", detected)
