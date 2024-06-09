import cv2
import numpy as np

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
