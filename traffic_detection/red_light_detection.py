import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 读取图像
image = cv2.imread('image\\2.png')
show_image('Original Image', image)

# 1. BGR空间转HSV空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
show_image('HSV Image', hsv)

# 2. 平滑滤波
smoothed = cv2.medianBlur(hsv, 5)
show_image('Smoothed Image', smoothed)

# 3. 高斯模糊
blurred = cv2.GaussianBlur(smoothed, (5, 5), 0)
show_image('Gaussian Blurred Image', blurred)

# 4. 设置HSV阈值提取红色，并创建mask掩盖非红色区域
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(blurred, lower_red1, upper_red1)
mask2 = cv2.inRange(blurred, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)
show_image('Red Mask', mask)

# 5. Canny边缘检测
edges = cv2.Canny(mask, 100, 200)
show_image('Canny Edges', edges)

# 6. 形态学操作，侵蚀和膨胀让红灯轮廓闭合
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
show_image('Morphologically Closed Edges', closed)

# 7. 设置阈值，过滤掉过小或过大得闭合区域
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [cnt for cnt in contours if 500 < cv2.contourArea(cnt) < 3000]

# 创建一个空图像来绘制筛选后的轮廓
filtered_contours_img = np.zeros_like(image)
cv2.drawContours(filtered_contours_img, filtered_contours, -1, (0, 255, 0), 3)
show_image('Filtered Contours', filtered_contours_img)

# 8. 形态学筛选出圆形得图形，并输出检测到红灯，返回True
detected = False
for cnt in filtered_contours:
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * cv2.contourArea(cnt) / (perimeter * perimeter)
    if 0.7 < circularity < 1.2:  # Circularity threshold
        detected = True
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)

show_image('Detected Red Lights', image)

print("Red light detected:", detected)
