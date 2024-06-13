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
import roslibpy


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

    # Connect to ROS
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()
    print("Is ROS connected?", client.is_connected)
    talker = roslibpy.Topic(client, "/detection_labels", "std_msgs/String")

    F = 35  # Example focal length in mm, adjust this value as necessary
    H = 100  # Example actual height of object in mm, adjust this value as necessary

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        img_batch, scale_ratio, pad_size = preprocess_img(frame)
        img_batch = np.expand_dims(img_batch, axis=0)

        infer_start = time.time()
        output = model.infer([img_batch])
        output = torch.tensor(output[0])
        boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])

        pred_all = boxout[0].numpy()
        scale_coords(
            cfg["input_shape"],
            pred_all[:, :4],
            frame.shape,
            ratio_pad=(scale_ratio, pad_size),
        )

        detected_labels = draw_bbox(pred_all, class_names, F, H)
        print(detected_labels)

        # Publish to ROS
        talker.publish(roslibpy.Message({"data": detected_labels}))

    cap.release()
    cv2.destroyAllWindows()
    client.terminate()
    print("ROS connection terminated.")


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
