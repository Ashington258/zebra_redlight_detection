import cv2

def main():
    # 打开默认摄像头（设备索引为0）
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 检查是否成功读取帧
        if not ret:
            print("无法读取帧")
            break

        # 显示帧
        cv2.imshow('Camera Test', frame)

        # 检查用户是否按下 'q' 键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
