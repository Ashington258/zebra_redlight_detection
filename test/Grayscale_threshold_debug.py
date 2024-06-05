import cv2

def on_trackbar_changed(value):
    pass

def process_image(value):
    global img, gray, blurred, thresh
    cv2.imshow("Original", img)
    
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 根据滑动条值进行阈值二值化
    _, binary = cv2.threshold(blurred, value, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("Processed", binary)

# 读取图像
img = cv2.imread('image\\1_gray.png')
cv2.namedWindow("Original")
cv2.namedWindow("Processed")

# 创建滑动条
cv2.createTrackbar('Threshold', 'Processed', 0, 255, on_trackbar_changed)

# 初始化滑动条位置
initial_thresh = 127
cv2.setTrackbarPos('Threshold', 'Processed', initial_thresh)

# 初始处理图像
process_image(initial_thresh)

while True:
    # 等待键盘输入，按 'q' 退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # 获取滑动条值
    thresh = cv2.getTrackbarPos('Threshold', 'Processed')
    
    # 处理图像
    process_image(thresh)

cv2.destroyAllWindows()
