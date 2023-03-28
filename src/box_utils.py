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

"""Bbox utils"""

import math
import itertools as it
import numpy as np
from src.model_utils.config import config


class GeneratDefaultBoxes():
    """
    Generate Default boxes for retinanet, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y, x, h, w].
    `self.default_boxes_ltrb` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """
    def __init__(self):
        scales = np.array([2 ** 0])
        anchor_size = np.array(config.anchor_size)
        self.default_boxes = []
        for idex, feature_size in enumerate(config.feature_size):
            base_size = anchor_size[idex]
            size1 = base_size*scales[0]
            all_sizes = []
            for aspect_ratio in config.aspect_ratios[idex]:
                w1, h1 = size1 * math.sqrt(aspect_ratio), size1 / math.sqrt(aspect_ratio)
                all_sizes.append((h1, w1))
            for i, j in it.product(range(feature_size), repeat=2):
                for h, w in all_sizes:
                    cx, cy = (j + 0.5) * config.steps[idex], (i + 0.5) * config.steps[idex]
                    # cx, cy = j * config.steps[idex], i * config.steps[idex]
                    self.default_boxes.append([cx, cy, h, w])

        def to_ltrb(cx, cy, h, w):
            h, w = h, w
            return cx - h / 2, cy - w / 2, cx + h / 2, cy + w / 2

        # For IoU calculation
        self.default_boxes_ltrb = np.array(tuple(to_ltrb(*i) for i in self.default_boxes), dtype='float32')
        self.default_boxes = np.array(self.default_boxes, dtype='float32')

default_boxes_ltrb = GeneratDefaultBoxes().default_boxes_ltrb
default_boxes = GeneratDefaultBoxes().default_boxes
num_level_cells_list = [1600, 400, 100, 25]

def nanodetplus_bboxes_encode(boxes, img_id):
    t_boxes = np.zeros((2125, 100, 4), dtype=np.float32)
    t_label = np.full((2125, 100,), 80, dtype=np.int64)
    t_mask = np.zeros((2125, 100,), dtype=np.float32)
    num_gt = boxes.shape[0]
    t_boxes[:, :num_gt] = boxes[:num_gt, :4]
    t_label[:, :num_gt] = boxes[:num_gt, 4]
    t_mask[:, :num_gt] = 1.0
    return t_boxes, t_label.astype(np.int32), default_boxes, t_mask, num_gt

def retinanet_bboxes_decode(boxes):
    """Decode predict boxes to [y, x, h, w]"""
    boxes_t = boxes.copy()
    default_boxes_t = default_boxes.copy()
    boxes_t[:, :2] = boxes_t[:, :2] * config.prior_scaling[0] * default_boxes_t[:, 2:] + default_boxes_t[:, :2]
    boxes_t[:, 2:4] = np.exp(boxes_t[:, 2:4] * config.prior_scaling[1]) * default_boxes_t[:, 2:4]

    bboxes = np.zeros((len(boxes_t), 4), dtype=np.float32)

    bboxes[:, [0, 1]] = boxes_t[:, [0, 1]] - boxes_t[:, [2, 3]] / 2
    bboxes[:, [2, 3]] = boxes_t[:, [0, 1]] + boxes_t[:, [2, 3]] / 2

    return np.clip(bboxes, 0, 1)


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

if __name__ == "__main__":
    print(default_boxes)
    print("!")