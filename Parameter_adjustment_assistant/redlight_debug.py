import cv2
import numpy as np

def nothing(x):
    pass

# 创建窗口和滑动条
cv2.namedWindow('image')
cv2.createTrackbar('smooth', 'image', 1, 20, nothing)
cv2.createTrackbar('blur', 'image', 1, 20, nothing)
cv2.createTrackbar('h_min', 'image', 0, 179, nothing)
cv2.createTrackbar('h_max', 'image', 0, 179, nothing)
cv2.createTrackbar('s_min', 'image', 0, 255, nothing)
cv2.createTrackbar('s_max', 'image', 0, 255, nothing)
cv2.createTrackbar('v_min', 'image', 0, 255, nothing)
cv2.createTrackbar('v_max', 'image', 0, 255, nothing)
cv2.createTrackbar('canny1', 'image', 100, 500, nothing)
cv2.createTrackbar('canny2', 'image', 200, 500, nothing)
cv2.createTrackbar('min_radius', 'image', 0, 100, nothing)
cv2.createTrackbar('max_radius', 'image', 0, 100, nothing)

# 读取图像
image = cv2.imread('image\\2.png')

while True:
    # 获取滑动条的当前值
    smooth = cv2.getTrackbarPos('smooth', 'image')
    blur = cv2.getTrackbarPos('blur', 'image')
    h_min = cv2.getTrackbarPos('h_min', 'image')
    h_max = cv2.getTrackbarPos('h_max', 'image')
    s_min = cv2.getTrackbarPos('s_min', 'image')
    s_max = cv2.getTrackbarPos('s_max', 'image')
    v_min = cv2.getTrackbarPos('v_min', 'image')
    v_max = cv2.getTrackbarPos('v_max', 'image')
    canny1 = cv2.getTrackbarPos('canny1', 'image')
    canny2 = cv2.getTrackbarPos('canny2', 'image')
    min_radius = cv2.getTrackbarPos('min_radius', 'image')
    max_radius = cv2.getTrackbarPos('max_radius', 'image')
    
    # 复制图像以防止修改原始图像
    frame = np.copy(image)
    
    # 1. 平滑滤波
    if smooth > 1:
        frame = cv2.medianBlur(frame, smooth)

    # 2. 高斯模糊
    if blur > 1:
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    
    # 3. 转换到 HSV 并提取红色区域
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([h_min, s_min, v_min])
    upper_red = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # 4. Canny 边缘检测
    edges = cv2.Canny(mask, canny1, canny2)
    
    # 5. 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 6. 查找轮廓并过滤
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:
            valid_contours.append(contour)
    
    # 7. 形态学筛选出圆形图形
    detected = False
    for contour in valid_contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if min_radius < radius < max_radius:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            detected = True
    
    # 显示处理后的图像
    cv2.imshow('image', frame)
    
    if detected:
        print("Red light detected")
    
    # 按 q 退出
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
