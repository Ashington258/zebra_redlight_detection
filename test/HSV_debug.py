import cv2
import numpy as np

def nothing(x):
    pass

def main():
    # 创建一个空白图像
    blank_image = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('Color Extraction')

    # 创建滑动条
    cv2.createTrackbar('Lower H', 'Color Extraction', 0, 255, nothing)
    cv2.createTrackbar('Upper H', 'Color Extraction', 0, 255, nothing)
    cv2.createTrackbar('Lower S', 'Color Extraction', 0, 255, nothing)
    cv2.createTrackbar('Upper S', 'Color Extraction', 0, 255, nothing)
    cv2.createTrackbar('Lower V', 'Color Extraction', 0, 255, nothing)
    cv2.createTrackbar('Upper V', 'Color Extraction', 0, 255, nothing)

    while True:
        # 读取图像
        image = cv2.imread('image\\1.png')

        # 将图像从BGR转换为HSV空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 获取滑动条的当前位置
        lower_h = cv2.getTrackbarPos('Lower H', 'Color Extraction')
        upper_h = cv2.getTrackbarPos('Upper H', 'Color Extraction')
        lower_s = cv2.getTrackbarPos('Lower S', 'Color Extraction')
        upper_s = cv2.getTrackbarPos('Upper S', 'Color Extraction')
        lower_v = cv2.getTrackbarPos('Lower V', 'Color Extraction')
        upper_v = cv2.getTrackbarPos('Upper V', 'Color Extraction')

        # 定义要提取的颜色范围（在HSV空间中）
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])

        # 创建掩码，在指定范围内保留颜色
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        # 对原始图像应用掩码
        extracted_color = cv2.bitwise_and(image, image, mask=mask)

        # 显示原始图像和提取的颜色
        cv2.imshow('Color Extraction', np.hstack([image, extracted_color]))

        # 按下 ESC 键退出循环
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
