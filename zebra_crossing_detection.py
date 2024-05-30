import cv2
import numpy as np

def process_frame(frame):
    # 1. 裁剪图像上半部分 30%
    height, width = frame.shape[:2]
    roi = frame[int(height*0.3):, :]
    cv2.imshow("ROI", roi)

    # 2. 边缘提取轮廓
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow("Edges", edges)

    # 3. 形态学操作，侵蚀和膨胀
    kernel = np.ones((5,5),np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Morphed", morphed)

    # 4. 提取闭合轮廓区域作为 ROI
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(morphed)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    cv2.imshow("Mask", mask)

    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
    cv2.imshow("ROI Masked", roi_masked)

    # 5. 灰度化
    gray_masked = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Masked", gray_masked)

    # 6. 平滑滤波
    smooth = cv2.bilateralFilter(gray_masked, 9, 75, 75)
    cv2.imshow("Smooth", smooth)

    # 7. 高斯模糊
    blurred = cv2.GaussianBlur(smooth, (5, 5), 0)
    cv2.imshow("Blurred", blurred)

    # 8. canny 边缘检测
    edges_blurred = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Edges Blurred", edges_blurred)

    # 9. hough 变化检测直线
    lines = cv2.HoughLinesP(edges_blurred, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # 10. 统计直线数量
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(lines) > 10:  # 阈值设为 10，根据需要调整
            cv2.putText(frame, 'Zebra Crossing Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Number of lines detected: {len(lines)}")
    else:
        print("No lines detected")

    cv2.imshow("Detected Lines", roi)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Processed Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
