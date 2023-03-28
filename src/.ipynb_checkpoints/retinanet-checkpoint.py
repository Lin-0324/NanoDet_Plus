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

"""retinanet based resnet."""

import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from src.ghost_pan import GhostBottleneck

def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.97,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

class Integral(nn.Cell):
    def __init__(self):
        super(Integral, self).__init__()
        self.softmax = P.Softmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.linspace = Tensor([[0, 1, 2, 3, 4, 5, 6, 7]], mstype.float32)
        self.matmul = P.MatMul(transpose_b=True)

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(x, (-1, 8))
        x = self.softmax(x)
        x = self.matmul(x, self.linspace)
        out_shape = x_shape[:-1] + (4,)
        x = self.reshape(x, out_shape)
        return x

# distance(l, t, r, b)
class Distance2bbox(nn.Cell):
    def __init__(self):
        super(Distance2bbox, self).__init__()
        self.stack = P.Stack(-1)

    def construct(self, points, distance):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        return self.stack([x1, y1, x2, y2])

    
# point(y, x)  bbox(y, x, y, x)
class BBox2Distance(nn.Cell):
    def __init__(self):
        super(BBox2Distance, self).__init__()
        self.stack = P.Stack(-1)

    def construct(self, points, bbox):
        left = points[..., 0] - bbox[..., 0]
        top = points[..., 1] - bbox[..., 1]
        right = bbox[..., 2] - points[..., 0]
        bottom = bbox[..., 3] - points[..., 1]
        # left = C.clip_by_value(left, Tensor(0.0), Tensor(6.9))
        # top = C.clip_by_value(top, Tensor(0.0), Tensor(6.9))
        # right = C.clip_by_value(right, Tensor(0.0), Tensor(6.9))
        # bottom = C.clip_by_value(bottom, Tensor(0.0), Tensor(6.9))
        return self.stack((left, top, right, bottom))


class QualityFocalLoss(nn.Cell):
    def __init__(self, beta=2.0, loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        self.sigmoid = P.Sigmoid()
        self.sigmiod_cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        self.pow = P.Pow()
        self.abs = P.Abs()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.reduce_sum = P.ReduceSum()
        self.be = beta
        self.loss_weight = loss_weight
        self.cast = P.Cast()

    def construct(self, logits, label, score, avg_factor):
        # print(logits[0])
        logits_sigmoid = self.sigmoid(logits)
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        score = self.tile(self.expand_dims(score, -1), (1, F.shape(logits)[-1]))
        label = label * score
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        modulating_factor = self.pow(self.abs(label - logits_sigmoid), self.be)
        qfl_loss = sigmiod_cross_entropy * modulating_factor
        # qfl_loss = qfl_loss * weight
        qfl_loss = self.reduce_sum(qfl_loss) / avg_factor
        loss = self.loss_weight * qfl_loss
        loss = self.cast(loss, ms.float32)
        return loss

class DistributionFocalLoss(nn.Cell):
    def __init__(self, loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.loss_weight = loss_weight
        self.reshape = P.Reshape()

    def construct(self, pred, label, weight, avg_factor):
        dis_left = self.cast(F.floor(label), mstype.int32)
        # print(dis_left)
        dis_right = dis_left + 1
        weight_left = self.cast(dis_right, mstype.float32) - label
        weight_right = label - self.cast(dis_left, mstype.float32)
        dfl_loss = self.cross_entropy(pred, dis_left) * weight_left + self.cross_entropy(pred, dis_right) * weight_right
        dfl_loss = self.reshape(dfl_loss, (-1,))
        dfl_loss = dfl_loss * weight
        dfl_loss = self.reduce_sum(dfl_loss) / avg_factor
        loss = self.loss_weight * dfl_loss
        loss = self.cast(loss, ms.float32)
        return loss

class GIou(nn.Cell):
    """Calculating giou"""

    def __init__(self, loss_weight=1.0):
        super(GIou, self).__init__()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.div = P.RealDiv()
        self.eps = 0.000001
        self.loss_weight = loss_weight
        self.reduce_sum = P.ReduceSum()

    def construct(self, box_p, box_gt, weight, avg_factor):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        xc_1 = self.min(box_p[..., 0:1], box_gt[..., 0:1])
        xc_2 = self.max(box_p[..., 2:3], box_gt[..., 2:3])
        yc_1 = self.min(box_p[..., 1:2], box_gt[..., 1:2])
        yc_2 = self.max(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        res_mid0 = c_area - union
        res_mid1 = self.div(self.cast(res_mid0, ms.float32), self.cast(c_area, ms.float32))
        giou = iou - res_mid1
        giou = C.clip_by_value(giou, -1.0, 1.0)
        giou = 1 - giou
        giou = self.reshape(giou, (-1,))
        giou = giou * weight
        giou = self.reduce_sum(giou) / avg_factor
        loss = self.loss_weight * giou
        loss = self.cast(loss, ms.float32)
        return loss

class Iou(nn.Cell):
    def __init__(self):
        super(Iou, self).__init__()
        self.cast = P.Cast()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])

        w = self.max(x_2 - x_1, F.scalar_to_array(0.0))
        h = self.max(y_2 - y_1, F.scalar_to_array(0.0))

        intersection = w * h
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        iou = C.clip_by_value(iou, 0.0, 1.0)
        iou = iou.squeeze(-1)
        return iou

class retinanetWithLossCell(nn.Cell):
    def __init__(self, network, config):
        super(retinanetWithLossCell, self).__init__()
        self.network = network
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.less = P.Less()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.sigmoid = P.Sigmoid()
        self.ones = P.Ones()
        self.iou = Iou()
        self.loss_bbox = GIou(loss_weight=2.0)
        self.loss_qfl = QualityFocalLoss(loss_weight=1.0)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.integral = Integral()
        self.distance2bbox = Distance2bbox()
        self.bbox2distance = BBox2Distance()
        self.sigmoid = P.Sigmoid()
        self.argmax = P.ArgMaxWithValue(axis=-1)
        self.max = P.Maximum()
        self.stack = P.Stack(-1)
        self.ones_like = P.OnesLike()
        self.loss_zero = Tensor(0.0, dtype=mstype.float32)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.sigmiod_cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        self.pow = P.Pow()
        self.abs = P.Abs()
        self.topk = 13
        self.batch_iter = Tensor(np.arange(0, config.batch_size * 80), mstype.int32)

    def construct(self, x, res_boxes, res_labels, res_center_priors, res_mask, nums_match):
        bbox_preds, cls_scores ,aux_bbox_preds, aux_cls_scores = self.network(x)    
        cls_scores = self.cast(cls_scores, mstype.float32)
        bbox_preds = self.cast(bbox_preds, mstype.float32)
        aux_bbox_preds = self.cast(aux_bbox_preds, mstype.float32)
        aux_cls_scores = self.cast(aux_cls_scores, mstype.float32)
        
        res_boxes = self.cast(res_boxes, mstype.float32)
        res_center_priors = self.cast(res_center_priors, mstype.float32)
        res_mask = self.cast(res_mask, mstype.float32)
        bbox_pred_corners = self.integral(bbox_preds) * self.tile(self.expand_dims(res_center_priors[..., 2], -1), (1, 1, 4))
        decode_bbox_pred = self.distance2bbox(res_center_priors[..., :2], bbox_pred_corners)
        aux_bbox_pred_corners = self.integral(aux_bbox_preds) * self.tile(self.expand_dims(res_center_priors[..., 2], -1), (1, 1, 4))
        aux_decode_bbox_pred = self.distance2bbox(res_center_priors[..., :2], aux_bbox_pred_corners)
        
        # N, 2125, 80, 4
        prior_center = res_center_priors[..., :2]
        lt_ = self.tile(self.expand_dims(prior_center, 2), (1, 1, 80, 1)) - res_boxes[..., :2]
        rb_ = res_boxes[..., 2:] - self.tile(self.expand_dims(prior_center, 2), (1, 1, 80, 1))
        deltas = F.concat((lt_, rb_), -1)
        
        # N, 2125, 80
        min_index, min_value = P.ArgMinWithValue(-1)(deltas)
        min_value = min_value * res_mask
        # print(min_value)
        deltas_ones = F.ones_like(min_value)
        
        # N, 2125, 80
        is_in_gts = (min_value > 0) * deltas_ones
        valid_mask = P.ReduceSum()(is_in_gts, -1) > 0
        valid_mask_tilt = self.tile(self.expand_dims(valid_mask, -1), (1, 1, 80))
        pairwise_ious = self.iou(self.tile(self.expand_dims(aux_decode_bbox_pred, 2), (1, 1, 80, 1)), res_boxes) * valid_mask_tilt
        iou_cost = -P.Log()(pairwise_ious + 1e-8)       
        onehot_labels = self.onehot(res_labels, F.shape(aux_cls_scores)[-1], self.on_value, self.off_value)
        aux_cls_scores_tile = self.tile(self.expand_dims(aux_cls_scores, 2), (1, 1, 80, 1))
        soft_labels = onehot_labels * self.tile(self.expand_dims(pairwise_ious, -1), (1, 1, 1, F.shape(aux_cls_scores)[-1]))
        scale_factor = soft_labels - self.sigmoid(aux_cls_scores_tile)
        modulating_factor = self.pow(self.abs(scale_factor), 2.0)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(aux_cls_scores_tile, soft_labels)
        cls_cost = sigmiod_cross_entropy * modulating_factor
        cls_cost = P.ReduceSum()(cls_cost, -1) * res_mask
        cost_matrix = cls_cost + iou_cost * 3.0
        matching_matrix = F.zeros_like(cost_matrix)
        pairwise_ious_topk = P.Transpose()(pairwise_ious, (0, 2, 1))
        topk_ious, _ = P.TopK(True)(pairwise_ious_topk, self.topk)
        topk_ious = P.Transpose()(topk_ious, (0, 2, 1))
        dynamic_ks = C.clip_by_value(self.cast(P.ReduceSum()(topk_ious, 1), mstype.int32), Tensor(1.0), Tensor(2124.0))
        dynamic_ks = self.cast(dynamic_ks, mstype.int32)
        dynamic_ks_indices = P.Stack(1)((self.batch_iter, dynamic_ks.reshape(-1, )))
        dynamic_ks_indices = F.stop_gradient(dynamic_ks_indices)
        
        cost_topk = P.Transpose()(cost_matrix, (0, 2, 1))
        values, _ = P.TopK(True)(- cost_topk, self.topk)
        values = self.reshape(-values, (-1, self.topk))
        # N, 80, 1
        max_neg_score = self.expand_dims(
            P.GatherNd()(values, dynamic_ks_indices).reshape(F.shape(aux_cls_scores)[0], -1), 2)
        # N, 2125, 80
        max_neg_score = P.Transpose()(max_neg_score, (0, 2, 1))
        pos_mask = F.cast(cost_matrix <= max_neg_score, mstype.float32)
        pos_mask = pos_mask * res_mask
        cost_t = cost_matrix * pos_mask + (1.0 - pos_mask) * 2000.
        min_index, min_value = P.ArgMinWithValue(axis=2)(cost_t)
        # N, 2125, 80
        go_indx = self.onehot(min_index, F.shape(res_mask)[2], self.on_value, self.off_value)
        pos_mask = go_indx * pos_mask
        pos_mask = F.stop_gradient(pos_mask)
        
        pos_mask = self.cast(pos_mask, mstype.float16)
        res_labels = self.cast(res_labels, mstype.float16)
        res_boxes = self.cast(res_boxes, mstype.float16)
        
        pos_res_labels = self.reshape(
            P.BatchMatMul()(self.expand_dims(pos_mask, 2), self.expand_dims(res_labels, -1)),
            (F.shape(aux_cls_scores)[0], 2125)) 
        pos_labels_mask = self.reshape(
            P.BatchMatMul()(self.expand_dims(pos_mask, 2), self.expand_dims(self.ones_like(pos_mask), -1)),
            (F.shape(aux_cls_scores)[0], 2125))
        pos_res_labels = pos_res_labels * pos_labels_mask + (1.0 - pos_labels_mask) * 80.
        pos_res_boxes = self.reshape(
            P.BatchMatMul()(self.expand_dims(pos_mask, 2), res_boxes),
            (F.shape(aux_cls_scores)[0], 2125, 4))
        num_total_samples = self.cast(self.reduce_sum(self.cast(pos_mask, mstype.float32)), mstype.float32)
        
        num_total_samples = F.stop_gradient(num_total_samples)
        pos_res_labels = F.stop_gradient(pos_res_labels)
        pos_labels_mask = F.stop_gradient(pos_labels_mask)
        pos_res_boxes = F.stop_gradient(pos_res_boxes)
        
        cls_scores = self.cast(cls_scores, mstype.float32)
        bbox_preds = self.cast(bbox_preds, mstype.float32)
        aux_bbox_preds = self.cast(aux_bbox_preds, mstype.float32)
        aux_cls_scores = self.cast(aux_cls_scores, mstype.float32)
        
        cls_scores = self.reshape(cls_scores, (-1, 80))
        bbox_preds = self.reshape(bbox_preds, (-1, 32))
        aux_cls_scores = self.reshape(aux_cls_scores, (-1, 80))
        aux_bbox_preds = self.reshape(aux_bbox_preds, (-1, 32))
        
        pos_res_boxes = self.reshape(pos_res_boxes, (-1, 4))
        pos_res_labels = self.reshape(pos_res_labels, (-1,))
        
        res_center_priors = self.reshape(res_center_priors, (-1, 4))
        mask = self.cast(self.less(pos_res_labels, 80), mstype.float32)
        mask_bbox = self.tile(self.expand_dims(mask, -1), (1, 4))
        
        weight_targets = self.argmax(self.sigmoid(aux_cls_scores))[1] * mask
        weight_ones = self.ones_like(weight_targets) * mask
        
        grid_cell_centers = res_center_priors
        aux_bbox_pred_corners = self.integral(aux_bbox_preds) * self.tile(
                self.expand_dims(grid_cell_centers[..., 2], -1), (1, 4))
        aux_decode_bbox_pred = self.distance2bbox(grid_cell_centers, aux_bbox_pred_corners)
        bbox_pred_corners = self.integral(bbox_preds) * self.tile(
                self.expand_dims(grid_cell_centers[..., 2], -1), (1, 4))
        decode_bbox_pred = self.distance2bbox(grid_cell_centers, bbox_pred_corners)
        decode_bbox_target = pos_res_boxes
        pos_res_labels = self.cast(pos_res_labels, mstype.int32)
        
        # loss_bbox
        aux_loss_bbox = self.loss_bbox(aux_decode_bbox_pred, decode_bbox_target, weight_ones, num_total_samples)
        loss_bbox = self.loss_bbox(decode_bbox_pred, decode_bbox_target, weight_ones, num_total_samples)
     
        # loss_dfl
        axu_pred_corners = self.reshape(aux_bbox_preds, (-1, 8))
        pred_corners = self.reshape(bbox_preds, (-1, 8))
        target_corners = self.reshape(self.bbox2distance(grid_cell_centers, decode_bbox_target) / self.tile(
                self.expand_dims(grid_cell_centers[..., 2], -1), (1, 1, 4)), (-1,))
        target_corners = F.stop_gradient(target_corners)
        target_corners = C.clip_by_value(target_corners, Tensor(0.0), Tensor(6.9))
        weight_targets_dfl = self.reshape(self.tile(self.expand_dims(weight_targets, -1), (1, 4)), (-1,))
        weight_dfl_ones = self.reshape(self.tile(self.expand_dims(weight_ones, -1), (1, 4)), (-1,))
        aux_loss_dfl = self.loss_dfl(axu_pred_corners, target_corners, weight_dfl_ones, num_total_samples)
        loss_dfl = self.loss_dfl(pred_corners, target_corners, weight_dfl_ones, num_total_samples)
        
        # loss_qfl
        aux_score = self.iou(aux_decode_bbox_pred, decode_bbox_target) * mask
        aux_score = F.stop_gradient(aux_score)
        score = self.iou(decode_bbox_pred, decode_bbox_target) * mask
        aux_loss_qlf = self.loss_qfl(aux_cls_scores, pos_res_labels, aux_score, num_total_samples)
        loss_qfl = self.loss_qfl(cls_scores, pos_res_labels, aux_score, num_total_samples)
    
        loss = loss_qfl + loss_bbox + loss_dfl + aux_loss_qlf + aux_loss_bbox + aux_loss_dfl
        
        return loss

    
class ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = 0
        conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='same',
                         padding=padding)
        layers = [conv, _bn(out_planes), nn.ReLU()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output

def _gn(channel):
    return nn.GroupNorm(num_groups=32, num_channels=channel)

class ConvGNLR(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvGNLR, self).__init__()
        self.padding = padding
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad',
                         padding=padding)
        layers = [conv, _gn(out_channels), nn.LeakyReLU(alpha=0.1)]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
    
class DepthwiseConvModule(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=2,
            use_aux=False,
    ):
        super(DepthwiseConvModule, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            pad_mode='pad',
            group=in_channels,
            has_bias=False,
            weight_init="he_uniform",
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            pad_mode='same',
            has_bias=False,
            weight_init="he_uniform")
        self.use_aux = use_aux
        self.dwnorm = nn.BatchNorm2d(in_channels)
        self.pwnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(alpha=0.1)

    def construct(self, x):
        x = self.depthwise(x)
        x = self.dwnorm(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pwnorm(x)
        x = self.act(x)
        return x

class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Args:
        config (dict): The default config of retinanet.

    Returns:
        Tensor, flatten predictions.
    """
    def __init__(self, config):
        super(FlattenConcat, self).__init__()
        self.num_retinanet_boxes = config.num_retinanet_boxes
        self.concat = P.Concat(axis=1)
        self.transpose = P.Transpose()
    def construct(self, inputs):
        output = ()
        batch_size = F.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (F.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return F.reshape(res, (batch_size, self.num_retinanet_boxes, -1))

def ClassificationModel(in_channel, num_anchors, kernel_size=3,
                        stride=1, pad_mod='same', num_classes=81, feature_size=96):
    conv2 = DepthwiseConvModule(feature_size, feature_size, kernel_size=3)
    conv3 = DepthwiseConvModule(feature_size, feature_size, kernel_size=3)
    conv5 = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, pad_mode='same',
                      has_bias=True,
                      weight_init="Normal",
                      bias_init=-4.595)
    return nn.SequentialCell([conv2, conv3,conv5])

def RegressionModel(in_channel, num_anchors, kernel_size=3, stride=1, pad_mod='same', feature_size=96):
    conv2 = DepthwiseConvModule(feature_size, feature_size, kernel_size=3)
    conv3 = DepthwiseConvModule(feature_size, feature_size, kernel_size=3)
    conv5 = nn.Conv2d(feature_size, 32, kernel_size=3, pad_mode='same',
                      has_bias=True,
                      weight_init="Normal",
                      )
    return nn.SequentialCell([conv2, conv3, conv5])

def ShareClassRegModel(in_channel, num_anchors, num_classes, feature_size=96):
    conv2 = DepthwiseConvModule(feature_size, feature_size, kernel_size=5, padding=5 // 2)
    conv3 = DepthwiseConvModule(feature_size, feature_size, kernel_size=5, padding=5 // 2)
    return nn.SequentialCell([conv2, conv3])


def ShareClassRegModelAUX(in_channel, feature_size=192):
    conv2 = ConvGNLR(in_channel, in_channel, kernel_size=3, padding=1)
    conv3 = ConvGNLR(in_channel, in_channel, kernel_size=3, padding=1)
    conv4 = ConvGNLR(in_channel, in_channel, kernel_size=3, padding=1)
    conv5 = ConvGNLR(in_channel, in_channel, kernel_size=3, padding=1)
    return nn.SequentialCell([conv2, conv3, conv4, conv5])


class MultiBox(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.

    Args:
        config (dict): The default config of retinanet.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, config):
        super(MultiBox, self).__init__()

        out_channels = config.extras_out_channels
        num_default = config.num_default
        loc_layers = []
        cls_layers = []
        share_layers = []
        for k, out_channel in enumerate(out_channels):
            # loc_layers += [RegressionModel(in_channel=out_channel, num_anchors=num_default[k])]
            # cls_layers += [ClassificationModel(in_channel=out_channel, num_anchors=num_default[k],
            #                                    num_classes=config.num_classes)]
            loc_layers += [nn.Conv2d(in_channels=out_channel, out_channels=32,
                                     kernel_size=1,
                                     has_bias=True,
                                     weight_init="Normal",
                                     )]
            cls_layers += [nn.Conv2d(in_channels=out_channel, out_channels=80,
                                     kernel_size=1,
                                     has_bias=True,
                                     weight_init="Normal",
                                     bias_init=-4.595)]
            share_layers += [ShareClassRegModel(in_channel=out_channel, num_anchors=num_default[k],
                                                num_classes=config.num_classes)]

        self.multi_share_layers = nn.layer.CellList(share_layers)
        self.multi_loc_layers = nn.layer.CellList(loc_layers)
        self.multi_cls_layers = nn.layer.CellList(cls_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        # share_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            # B, C, H, W
            share_output = self.multi_share_layers[i](inputs[i])
            loc_outputs += (self.multi_loc_layers[i](share_output),)
            cls_outputs += (self.multi_cls_layers[i](share_output),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)

class MultiBoxAUX(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.

    Args:
        config (dict): The default config of retinanet.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, config):
        super(MultiBoxAUX, self).__init__()

        out_channels = config.extras_out_channels_aux
        num_default = config.num_default
        loc_layers = []
        cls_layers = []
        share_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [nn.Conv2d(in_channels=out_channel, out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     pad_mode="pad",
                                     has_bias=False,
                                     weight_init="Normal")]
            cls_layers += [nn.Conv2d(in_channels=out_channel, out_channels=config.num_classes,
                                     kernel_size=3,
                                     padding=1,
                                     pad_mode="pad",
                                     has_bias=True,
                                     weight_init="Normal",
                                     bias_init=-4.595)]
            share_layers += [ShareClassRegModelAUX(in_channel=out_channel)]

        self.multi_loc_layers = nn.layer.CellList(loc_layers)
        self.multi_cls_layers = nn.layer.CellList(cls_layers)
        self.multi_share_layers = nn.layer.CellList(share_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            share_output = self.multi_share_layers[i](inputs[i])
            loc_outputs += (self.multi_loc_layers[i](share_output),)
            cls_outputs += (self.multi_cls_layers[i](share_output),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)
    
class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of retinanet network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss

class ShuffleV2Block(nn.Cell):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        branch_features = oup // 2
        if self.stride > 1:
            self.branch1 = nn.SequentialCell([
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, padding=0, pad_mode='pad', has_bias=False),
                nn.BatchNorm2d(branch_features),
                nn.LeakyReLU(alpha=0.1),
            ])
        else:
            self.branch1 = nn.SequentialCell()

        self.branch2 = nn.SequentialCell([
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(alpha=0.1),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(alpha=0.1),
        ])

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, "pad", padding, group=i, has_bias=bias)

    def construct(self, x):
        if self.stride == 1:
            x1, x2 = P.Split(axis=1, output_num=2)(x)
            out = P.Concat(axis=1)((x1, self.branch2(x2)))
        else:
            out = P.Concat(axis=1)((self.branch1(x), self.branch2(x)))
        out = channel_shuffle(out, 2)
        return out

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = P.Reshape()(x, (batchsize, groups, channels_per_group, height, width))
    x = P.Transpose()(x, (0, 2, 1, 3, 4))
    x = P.Reshape()(x, (batchsize, -1, height, width))
    return x

class ShuffleNetV2(nn.Cell):
    def __init__(self, model_size='1.0x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        elif model_size == '3.0x':
            self.stage_out_channels = [-1, 24, 512, 1024, 2048, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(num_features=input_channel, momentum=0.9),
            nn.LeakyReLU(alpha=0.1),
        ])

        self.pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), "CONSTANT")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)


        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            feature = []
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    feature.append(ShuffleV2Block(input_channel, output_channel, stride=2))
                else:
                    feature.append(ShuffleV2Block(output_channel, output_channel, stride=1))

                input_channel = output_channel
            self.features.append(feature)

        # self.features = nn.SequentialCell([*self.features])

        self.stage2 = nn.SequentialCell([*self.features[0]])
        self.stage3 = nn.SequentialCell([*self.features[1]])
        self.stage4 = nn.SequentialCell([*self.features[2]])
        self._initialize_weights()

    def construct(self, x):
        x = self.conv1(x)
        x = self.pad(x)
        x = self.maxpool(x)
        C2 = self.stage2(x)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        return C2, C3, C4

    def _initialize_weights(self):
        param_dict = load_checkpoint("./shufflenetV2_x1.ckpt")
        load_param_into_net(self, param_dict)
        print("shufflenetV2 init done!")

def resnet50(num_classes):
    return ShuffleNetV2(model_size='1.0x')

class GhostBlocks(nn.Cell):
    """Stack of GhostBottleneck used in GhostPAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        activation (str): Name of activation function. Default: LeakyReLU.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            expand=1,
            kernel_size=5,
            num_blocks=1,
            use_res=False,
            activation="LeakyReLU",
    ):
        super(GhostBlocks, self).__init__()
        self.use_res = use_res
        # if use_res:
        #     self.reduce_conv = ConvModule(
        #         in_channels,
        #         out_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         activation=activation,
        #     )
        if use_res:
            self.reduce_conv = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(alpha=0.1)]
            )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                GhostBottleneck(
                    in_channels,
                    int(out_channels * expand),
                    out_channels,
                    kernel_size=kernel_size,
                    act_type=activation,
                )
            )
        self.blocks = nn.SequentialCell(*[blocks])

    def construct(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out


class retinanet50(nn.Cell):
    def __init__(self, backbone, config, is_training=True):
        super(retinanet50, self).__init__()

        self.backbone = backbone
        feature_size = config.feature_size
        self.concat = P.Concat(axis=1)
        self.P5_1 = nn.Conv2d(464, 96, kernel_size=1, stride=1, pad_mode='same',
                              has_bias=False,
                              weight_init="xavier_uniform")
        self.P5_norm = nn.BatchNorm2d(num_features=96)
        self.P5_leakyReLU = nn.LeakyReLU(alpha=0.1)
        self.P_upsample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        self.P5_2 = GhostBlocks(192, 96)
        self.P5_3 = GhostBlocks(192, 96)
        self.P4_1 = nn.Conv2d(232, 96, kernel_size=1, stride=1, pad_mode='same',
                              has_bias=False,
                              weight_init="xavier_uniform")
        self.P4_norm = nn.BatchNorm2d(num_features=96)
        self.P4_leakyReLU = nn.LeakyReLU(alpha=0.1)
        self.P_upsample2 = P.ResizeBilinear((feature_size[0], feature_size[0]), half_pixel_centers=True)
        self.P4_2 = GhostBlocks(in_channels=192, out_channels=96)
        self.P4_3 = GhostBlocks(in_channels=192, out_channels=96)

        self.P3_1 = nn.Conv2d(116, 96, kernel_size=1, stride=1, pad_mode='same',
                              has_bias=False,
                              weight_init="xavier_uniform")
        self.P3_norm = nn.BatchNorm2d(num_features=96)
        self.P3_leakyReLU = nn.LeakyReLU(alpha=0.1)
        self.P3_2 = GhostBlocks(192, 96)

        self.P3_downsample = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)
        self.P4_downsample = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)

        self.extra_lvl_in = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)
        self.extra_lvl_out = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)

        self.D5_1 = nn.Conv2d(464, 96, kernel_size=1, stride=1, pad_mode='same',
                              has_bias=False,
                              weight_init="xavier_uniform")
        self.D5_norm = nn.BatchNorm2d(num_features=96)
        self.D5_leakyReLU = nn.LeakyReLU(alpha=0.1)
        self.D_upsample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        self.D5_2 = GhostBlocks(192, 96)
        self.D5_3 = GhostBlocks(192, 96)
        self.D4_1 = nn.Conv2d(232, 96, kernel_size=1, stride=1, pad_mode='same',
                              has_bias=False,
                              weight_init="xavier_uniform")
        self.D4_norm = nn.BatchNorm2d(num_features=96)
        self.D4_leakyReLU = nn.LeakyReLU(alpha=0.1)
        self.D_upsample2 = P.ResizeBilinear((feature_size[0], feature_size[0]), half_pixel_centers=True)
        self.D4_2 = GhostBlocks(in_channels=192, out_channels=96)
        self.D4_3 = GhostBlocks(in_channels=192, out_channels=96)

        self.D3_1 = nn.Conv2d(116, 96, kernel_size=1, stride=1, pad_mode='same',
                              has_bias=False,
                              weight_init="xavier_uniform")
        self.D3_norm = nn.BatchNorm2d(num_features=96)
        self.D3_leakyReLU = nn.LeakyReLU(alpha=0.1)
        self.D3_2 = GhostBlocks(192, 96)

        self.D3_downsample = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)
        self.D4_downsample = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)

        self.extra_lvl_in_aux = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)
        self.extra_lvl_out_aux = DepthwiseConvModule(96, 96, kernel_size=5, stride=2, padding=2)
        self.multi_box = MultiBox(config)
        self.multi_box_aux = MultiBoxAUX(config)
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        C3, C4, C5 = self.backbone(x)
        # Ghost PAN
        # top -> down
        P5 = self.P5_1(C5)
        P5 = self.P5_norm(P5)
        P5 = self.P5_leakyReLU(P5)
        extra_lvl_in = self.extra_lvl_in(P5)
        P5_upsampled = self.P_upsample1(P5)

        P4 = self.P4_1(C4)
        P4 = self.P4_norm(P4)
        P4 = self.P4_leakyReLU(P4)
        P4 = self.concat([P4, P5_upsampled])
        P4 = self.P4_2(P4)
        P4_upsampled = self.P_upsample2(P4)

        P3 = self.P3_1(C3)
        P3 = self.P3_norm(P3)
        P3 = self.P3_leakyReLU(P3)
        P3 = self.concat([P3, P4_upsampled])
        P3 = self.P3_2(P3)

        # down -> top
        P3_downSampled = self.P3_downsample(P3)
        P4 = self.concat([P3_downSampled, P4])
        P4 = self.P4_3(P4)

        P4_downsampled = self.P4_downsample(P4)
        P5 = self.concat([P4_downsampled, P5])
        P5 = self.P5_3(P5)

        extra_lvl_out = self.extra_lvl_out(P5)
        extra_main = extra_lvl_in + extra_lvl_out
        multi_feature = (P3, P4, P5, extra_main)
        pred_loc, pred_label = self.multi_box(multi_feature)
        
        # 辅助训练模块
        # top -> down
        D5 = self.D5_1(C5)
        D5 = self.D5_norm(D5)
        D5 = self.D5_leakyReLU(D5)
        extra_lvl_in_aux = self.extra_lvl_in_aux(D5)
        D5_upsampled = self.D_upsample1(D5)

        D4 = self.D4_1(C4)
        D4 = self.D4_norm(D4)
        D4 = self.D4_leakyReLU(D4)
        D4 = self.concat([D4, D5_upsampled])
        D4 = self.D4_2(D4)
        D4_upsampled = self.D_upsample2(D4)

        D3 = self.D3_1(C3)
        D3 = self.D3_norm(D3)
        D3 = self.D3_leakyReLU(D3)
        D3 = self.concat([D3, D4_upsampled])
        D3 = self.D3_2(D3)

        # down -> top
        D3_downSampled = self.D3_downsample(D3)
        D4 = self.concat([D3_downSampled, D4])
        D4 = self.D4_2(D4)

        D4_downsampled = self.D4_downsample(D4)
        D5 = self.concat([D4_downsampled, D5])
        D5 = self.D5_2(D5)

        extra_lvl_out_aux = self.extra_lvl_out_aux(D5)
        extra_aux = extra_lvl_in_aux + extra_lvl_out_aux

        # 两个GhostPAN concat起来
        extra = self.concat([extra_main, extra_aux])
        E5 = self.concat([P5, D5])
        E4 = self.concat([P4, D4])
        E3 = self.concat([P3, D3])

        multi_feature_aux = (E3, E4, E5, extra)
        pred_loc_aux, pred_label_aux = self.multi_box_aux(multi_feature_aux)
        return pred_loc, pred_label, pred_loc_aux, pred_label_aux
    
    
class retinanetInferWithDecoder(nn.Cell):
    """
    retinanet Infer wrapper to decode the bbox locations.

    Args:
        network (Cell): the origin retinanet infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): retinanet config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.

    """
    def __init__(self, network, default_boxes, config):
        super(retinanetInferWithDecoder, self).__init__()
        self.network = network
        # self.distance2bbox = Distance2bbox(config.img_shape)
        self.distribution_project = Integral()
        self.center_priors = default_boxes
        self.sigmoid = P.Sigmoid()
        self.expandDim = P.ExpandDims()
        self.tile = P.Tile()
        self.shape = P.Shape()
        self.stack = P.Stack(-1)

    def construct(self, x, max_shape=None):
        x_shape = self.shape(x)
        default_priors = self.expandDim(self.center_priors, 0)
        bbox_preds, cls_scores ,aux_bbox_preds, aux_cls_scores = self.network(x)
        dis_preds = self.distribution_project(bbox_preds) * self.tile(self.expandDim(default_priors[..., 3], -1),
                                                                     (1, 1, 4))
        bboxes = self.distance2bbox(default_priors[..., :2], dis_preds, max_shape)
        scores = self.sigmoid(cls_scores)
        # scores = cls_preds
        return bboxes, scores

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        if max_shape is not None:
            x1 = C.clip_by_value(x1, Tensor(0.0), Tensor(320.0))
            y1 = C.clip_by_value(y1, Tensor(0.0), Tensor(320.0))
            x2 = C.clip_by_value(x2, Tensor(0.0), Tensor(320.0))
            y2 = C.clip_by_value(y2, Tensor(0.0), Tensor(320.0))
        return self.stack([x1, y1, x2, y2])

if __name__ == "__main__":
    from src.model_utils.config import config
    gt = Tensor([[40,30,200,300]],mstype.float32)
    box = Tensor([[100,70,150,170]],mstype.float32)
    iou = Iou()
    giou = GIou()

    iou_out = iou(box,gt)
    giou_out = giou(box, gt)


    x = Tensor(np.random.rand(1,3,320,320), mstype.float32)
    backbone = resnet50(81)
    net = retinanet50(backbone, config, True)
    out = net(x)
    print("!")