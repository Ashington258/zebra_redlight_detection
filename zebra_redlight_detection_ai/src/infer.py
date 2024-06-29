import os
import time
import cv2
import json
import yaml
import argparse
import numpy as np
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from ais_bench.infer.interface import InferSession
from det_utils import letterbox, scale_coords, nms, xyxy2xywh


def read_class_names(ground_truth_json):
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
    """
    根据检测结果绘制边界框。

    参数:
    bbox: 一个二维数组，每一行代表一个检测到的物体，包含坐标信息和置信度。
    names: 一个类ID到类名的映射列表。
    F: 相机的焦距。
    H: 摄像头的高度。

    返回值:
    一个字符串，包含每个检测到的物体的名称、置信度和与摄像头的距离。
    """
    # 初始化用于存储检测结果字符串的变量
    det_result_str = ""
    # 遍历每个检测到的物体
    for idx, class_id in enumerate(bbox[:, 5]):
        # 如果置信度低于0.05，则跳过当前物体
        if float(bbox[idx][4]) < float(0.05):
            continue
        # 计算边界框的高度
        bbox_height = bbox[idx][3] - bbox[idx][1]
        # 根据相机参数和边界框高度计算物体到摄像头的距离
        distance = (H * F) / bbox_height
        # 格式化并添加物体的名称、置信度和距离到结果字符串
        det_result_str += "{} {:.4f} distance: {:.2f} mm\n".format(
            names[int(class_id)], bbox[idx][4], distance
        )
    # 返回结果字符串
    return det_result_str


def preprocess_img(img):
    img_padded, scale_ratio, pad_size = letterbox(img, new_shape=[640, 640])
    img_padded = img_padded[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img_padded = img_padded / 255.0
    img_padded = np.ascontiguousarray(img_padded).astype(np.float16)
    return img_padded, scale_ratio, pad_size


def clear_camera_buffer(cap, timeout=1.0):
    """清空摄像头缓冲区"""
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


def main():
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

    while True:
        if State == "detection":
            print("Detection mode ongoing")
            frame = clear_camera_buffer(cap)  # 清空摄像头缓冲区
            if frame is None:
                break
            # ret, frame = cap.read()
            # if not ret:
            #     print("Error: Failed to capture image.")
            #     break
            print("Captured a new frame")  # 添加日志

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

                start_time = time.time()
                end_time = start_time + 3
                while (
                    time.time() < end_time
                ):  # 持续打印检测相同标签三秒钟，等价于停车三秒钟
                    print(detected_labels)
                    # time.sleep(0.1)  # 小睡眠以避免打印过于频繁

                State = "cool"  # 将状态机置为冷却状态
                detected_labels = []  # 清空检测到的标签

        if State == "cool":
            print("Cool mode ongoing")
            time.sleep(2)
            State = "detection"  # 再次进入detection状态

    cap.release()
    cv2.destroyAllWindows()


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
