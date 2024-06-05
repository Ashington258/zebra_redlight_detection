import cv2

def print_camera_properties(cap):
    # 获取并打印摄像头的各种属性
    properties = {
        "CAP_PROP_FRAME_WIDTH": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "CAP_PROP_FRAME_HEIGHT": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "CAP_PROP_FPS": cap.get(cv2.CAP_PROP_FPS),
        "CAP_PROP_BRIGHTNESS": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "CAP_PROP_CONTRAST": cap.get(cv2.CAP_PROP_CONTRAST),
        "CAP_PROP_SATURATION": cap.get(cv2.CAP_PROP_SATURATION),
        "CAP_PROP_HUE": cap.get(cv2.CAP_PROP_HUE),
        "CAP_PROP_GAIN": cap.get(cv2.CAP_PROP_GAIN),
        "CAP_PROP_EXPOSURE": cap.get(cv2.CAP_PROP_EXPOSURE),
        "CAP_PROP_AUTO_EXPOSURE": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "CAP_PROP_AUTOFOCUS": cap.get(cv2.CAP_PROP_AUTOFOCUS)
    }
    for prop, value in properties.items():
        print(f"{prop}: {value}")

def main():
    # 打开默认摄像头（设备索引为0）
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 打印摄像头属性
    print_camera_properties(cap)

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
        if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 键退出
            break

    # 释放摄像头并关闭所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
