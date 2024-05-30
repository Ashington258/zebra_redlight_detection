import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def zebra_crossing_detection(image_path, line_threshold=10):
    # 1. 读取图像
    image = cv2.imread(image_path)
    show_image('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 2. 裁剪去除图像上半部分30%
    height = image.shape[0]
    cropped_image = image[int(0.3 * height):, :]
    show_image('Cropped Image', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    
    # 3. 灰度化
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    show_image('Gray Image', gray)
    
    # 4. 平滑滤波
    blurred = cv2.medianBlur(gray, 5)
    show_image('Median Blurred Image', blurred)
    
    # 5. 高斯模糊
    gaussian_blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    show_image('Gaussian Blurred Image', gaussian_blurred)
    
    # 6. Canny 边缘检测
    edges = cv2.Canny(gaussian_blurred, 50, 150)
    show_image('Canny Edges', edges)
    
    # 7. 形态学操作，侵蚀和膨胀
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    show_image('Morphologically Closed Edges', closed_edges)
    
    # 8. 提取闭合轮廓区域作为 ROI，创造 mask 去除以外的其他区域
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed_edges)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    show_image('Mask with Contours', mask)
    
    roi = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    show_image('ROI Image', cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    
    # 9. 对 ROI 进行 Canny 边缘检测
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_edges = cv2.Canny(roi_gray, 50, 150)
    show_image('ROI Canny Edges', roi_edges)
    
    # 10. Hough 变化检测直线
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    # 11. 画出检测到的直线
    line_image = np.copy(cropped_image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    show_image('Detected Lines', cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    
    # 12. 判断是否检测到斑马线
    detected = lines is not None and len(lines) > line_threshold
    
    return detected

# 调用函数进行测试
image_path = 'image\\1.png'
detected = zebra_crossing_detection(image_path)
print("Zebra Crossing Detected: ", detected)
