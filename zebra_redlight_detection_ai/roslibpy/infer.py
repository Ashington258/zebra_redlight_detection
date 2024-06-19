import os
import time
import cv2
import json
import yaml
import argparse
import numpy as np
import roslibpy
import threading
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from ais_bench.infer.interface import InferSession
from det_utils import letterbox, scale_coords, nms, xyxy2xywh


def read_class_names(ground_truth_json):
    """
    Reads the class names from a ground truth JSON file and returns a dictionary mapping class IDs to class names.

    Parameters:
        ground_truth_json (str): The path to the ground truth JSON file.

    Returns:
        dict: A dictionary mapping class IDs to class names.

    Raises:
        FileNotFoundError: If the ground truth JSON file does not exist.
        JSONDecodeError: If the ground truth JSON file cannot be decoded.
    """
    with open(ground_truth_json, "r") as file:
        content = file.read()
    content = json.loads(content)
    categories = content.get("categories")
    names = {}
    for id, category in enumerate(categories):
        category_name = category.get("name")
        if len(category_name.split()) == 2:
            temp = category_name.split()
            category_name = temp[0] + "_" + temp[1]
        names[id] = category_name.strip("\n")
    return names


def draw_bbox(bbox, names, F, H):
    det_result_str = ""
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4]) < float(0.05):
            continue
        bbox_height = bbox[idx][3] - bbox[idx][1]
        distance = (H * F) / bbox_height
        det_result_str += "{} {:.4f} distance: {:.2f} mm\n".format(
            names[int(class_id)], bbox[idx][4], distance
        )
    return det_result_str


def preprocess_img(img):
    img_padded, scale_ratio, pad_size = letterbox(img, new_shape=[640, 640])
    img_padded = img_padded[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img_padded = img_padded / 255.0
    img_padded = np.ascontiguousarray(img_padded).astype(np.float16)
    return img_padded, scale_ratio, pad_size


def clear_camera_buffer(cap, timeout=0.5):
    """清空摄像头缓冲区"""
    # 该函数尤其重要，用于清除缓存区，如不执行此步，则后面读取视频帧函数会多次进入，因为Video.Capture.read()会抓取多帧
    # 清除方法，读取即清除
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        # 如果读取时间超过timeout秒，说明缓冲区已经清空
        if time.time() - start_time > timeout:
            break
    return frame


def ros_publisher(talker, pause_event):
    """ROS消息发布线程"""
    while True:
        pause_event.wait()  # Wait until the pause_event is set
        talker.publish(roslibpy.Message({"data": ""}))
        time.sleep(0.1)


def main():
    # Connect to ROS
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()
    print("Is ROS connected?", client.is_connected)
    talker = roslibpy.Topic(client, "/detection_labels", "std_msgs/String")

    # # Create a threading event to control the ROS publisher thread
    # pause_event = threading.Event()
    # pause_event.set()  # Initially set the event to start publishing
    # NOTE 多线程被我关闭了
    # # 启动ROS消息发布线程
    # threading.Thread(
    #     target=ros_publisher, args=(talker, pause_event), daemon=True
    # ).start()

    args = parse_args()
    cfg = {
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "input_shape": [640, 640],
    }
    yaml_file = "data.yaml"
    with open(yaml_file, errors="ignore") as f:
        yaml_data = yaml.safe_load(f)

    class_names = yaml_data["names"]
    model = InferSession(args.device_id, args.model)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    F = 35  # Example focal length in mm, adjust this value as necessary
    H = 100  # Example actual height of object in mm, adjust this value as necessary

    State = "detection"
    car_running_command = ["car running command"]
    while True:
        # 在主循环中增加发布空消息，让ROS程序正常运行,ROS系统必须需要一直接收空消息才能进入回调函数，改变标志位
        talker.publish(roslibpy.Message({"data": car_running_command}))

        if State == "detection":
            print("Detection mode ongoing")
            frame = clear_camera_buffer(cap)  # 清空摄像头缓冲区
            if frame is None:
                break

            img_batch, scale_ratio, pad_size = preprocess_img(frame)
            img_batch = np.expand_dims(img_batch, axis=0)

            output = model.infer([img_batch])
            output = torch.tensor(output[0])
            boxout = nms(
                output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"]
            )

            pred_all = boxout[0].numpy()
            scale_coords(
                cfg["input_shape"],
                pred_all[:, :4],
                frame.shape,
                ratio_pad=(scale_ratio, pad_size),
            )

            detected_labels = draw_bbox(pred_all, class_names, F, H)

            if len(detected_labels) != 0:  # 如果检测到标签，即字符串非空
                # pause_event.clear()  # Pause the ROS publisher thread
                start_time = time.time()
                end_time = start_time + 3
                while (
                    time.time() < end_time
                ):  # 持续打印检测相同标签三秒钟，等价于停车三秒钟
                    talker.publish(
                        roslibpy.Message({"data": detected_labels})
                    )  # 发布检测标签
                    print(detected_labels)  #
                    time.sleep(0.1)  # 小睡眠以避免过于频繁发布
                # pause_event.set()  # Resume the ROS publisher thread

                State = "cool"  # 将状态机置为冷却状态
                detected_labels = []  # 清空检测到的标签
            else:  #
                talker.publish(
                    roslibpy.Message({"data": car_running_command})
                )  # 发布检测标签

        if State == "cool":
            print("Cool mode ongoing")  # 发送两秒的空消息，不能响应
            detected_labels = []  # 清空检测到的标签
            start_time = time.time()
            end_time = start_time + 3
            while time.time() < end_time:
                talker.publish(
                    roslibpy.Message({"data": car_running_command})
                )  # 发布检测标签
                time.sleep(0.1)  # 小睡眠以避免过于频繁发布
            State = "detection"  # 再次进入detection状态

    cap.release()
    cv2.destroyAllWindows()
    client.terminate()


def parse_args():
    parser = argparse.ArgumentParser(description="YoloV5 offline model inference.")
    parser.add_argument(
        "--ground_truth_json",
        type=str,
        default="test/test.json",
        help="annotation file path",
    )
    parser.add_argument(
        "--model", type=str, default="output/yolov5s.om", help="om model path"
    )
    parser.add_argument("--device-id", type=int, default=0, help="device id")
    parser.add_argument("--output-dir", type=str, default="output", help="output path")
    parser.add_argument("--eval", action="store_true", help="compute mAP")
    parser.add_argument(
        "--visible",
        action="store_true",
        help="draw detect result at image and save to output/img",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
