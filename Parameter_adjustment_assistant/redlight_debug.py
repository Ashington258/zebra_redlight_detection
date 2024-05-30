import cv2
import numpy as np

def nothing(x):
    pass

# 创建窗口
cv2.namedWindow('Parameters')

# 创建滑动条
cv2.createTrackbar('Blur', 'Parameters', 1, 10, nothing)
cv2.createTrackbar('GaussianBlur', 'Parameters', 1, 10, nothing)
cv2.createTrackbar('LowH', 'Parameters', 0, 179, nothing)
cv2.createTrackbar('HighH', 'Parameters', 10, 179, nothing)
cv2.createTrackbar('LowS', 'Parameters', 100, 255, nothing)
cv2.createTrackbar('HighS', 'Parameters', 255, 255, nothing)
cv2.createTrackbar('LowV', 'Parameters', 100, 255, nothing)
cv2.createTrackbar('HighV', 'Parameters', 255, 255, nothing)
cv2.createTrackbar('MinRadius', 'Parameters', 10, 100, nothing)
cv2.createTrackbar('MaxRadius', 'Parameters', 20, 200, nothing)

def process_image(image_path):
    frame = cv2.imread(image_path)

    while True:
        # 读取滑动条的值
        blur = cv2.getTrackbarPos('Blur', 'Parameters')
        gaussian_blur = cv2.getTrackbarPos('GaussianBlur', 'Parameters')
        lowH = cv2.getTrackbarPos('LowH', 'Parameters')
        highH = cv2.getTrackbarPos('HighH', 'Parameters')
        lowS = cv2.getTrackbarPos('LowS', 'Parameters')
        highS = cv2.getTrackbarPos('HighS', 'Parameters')
        lowV = cv2.getTrackbarPos('LowV', 'Parameters')
        highV = cv2.getTrackbarPos('HighV', 'Parameters')
        min_radius = cv2.getTrackbarPos('MinRadius', 'Parameters')
        max_radius = cv2.getTrackbarPos('MaxRadius', 'Parameters')

        # 平滑滤波
        if blur > 0:
            frame = cv2.medianBlur(frame, blur * 2 + 1)

        # 高斯模糊
        if gaussian_blur > 0:
            frame = cv2.GaussianBlur(frame, (gaussian_blur * 2 + 1, gaussian_blur * 2 + 1), 0)

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 设置HSV阈值提取红色
        lower_red = np.array([lowH, lowS, lowV])
        upper_red = np.array([highH, highS, highV])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Canny边缘检测
        edges = cv2.Canny(mask, 50, 150)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if min_radius <= radius <= max_radius:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                detected = True

        # 显示处理后的图像
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Edges', edges)

        if detected:
            print("Red light detected")

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# 传入待检测的图像路径
image_path = 'path_to_your_image.jpg'
process_image(image_path)
