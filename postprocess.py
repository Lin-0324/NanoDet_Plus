# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Evaluation for NanoDet"""

import os
import numpy as np
from PIL import Image
from src.warp import ShapeTransform
from src.coco_eval import metrics
from src.model_utils.config import config


def get_pred(result_path, img_id):
    boxes_file = os.path.join(result_path, img_id + '_0.bin')
    scores_file = os.path.join(result_path, img_id + '_1.bin')
    # scores = 1
    boxes = np.fromfile(boxes_file, dtype=np.float32).reshape(2125, 4)
    # boxes = np.fromfile(boxes_file, dtype=np.float32).reshape(3, 320, 320)
    # print("boxes",boxes)
    scores = np.fromfile(scores_file, dtype=np.float32).reshape(2125, 80)
    return boxes, scores


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size


def get_img_id(img_id_file):
    f = open(img_id_file)
    lines = f.readlines()

    ids = []
    for line in lines:
        ids.append(int(line))

    return ids

def get_warp_matrix(file_name):
    image = Image.open(file_name)
    image = np.array(image)
    pipline = ShapeTransform(keep_ratio=False)
    meta_data = dict(
        img=image,
    )
    meta = pipline(meta_data, dst_shape=(320, 320))
    return meta["warp_matrix"]

def cal_acc(result_path, img_path, img_id_file):
    ids = get_img_id(img_id_file)
    imgs = os.listdir(img_path)
    pred_data = []

    for img in imgs:
        img_id = img.split('.')[0]
        if int(img_id) not in ids:
            continue
        boxes, box_scores = get_pred(result_path, img_id)
        w, h = get_img_size(os.path.join(img_path, img))
        img_shape = np.array((h, w), dtype=np.float32)
        M = get_warp_matrix(os.path.join(img_path, img))
        pred_data.append({"boxes": boxes,
                          "box_scores": box_scores,
                          "img_id": int(img_id),
                          "image_shape": img_shape,
                          "warp_matrix": M})

    mAP = metrics(pred_data)
    print(f"mAP: {mAP}")


if __name__ == '__main__':
    cal_acc(config.result_path, config.img_path, config.img_id_file)

