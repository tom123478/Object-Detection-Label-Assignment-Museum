# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from tools import bbox_iou, probiou, xywhr2xyxyxyxy


class TaskAlignedAssigner(nn.Module):
    """
    用于目标检测的任务对齐分配器。

    此类基于任务对齐的度量，结合分类和定位信息，将真实目标（ground-truth objects）分配给锚点。

    属性：
        topk (int): 考虑的候选目标的数量。
        num_classes (int): 对象类别的总数。
        alpha (float): 用于分类部分的任务对齐度量的 alpha 参数。
        beta (float): 用于定位部分的任务对齐度量的 beta 参数。
        eps (float): 防止除零的小值。
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """使用可自定义的超参数初始化 TaskAlignedAssigner 对象。"""
        super().__init__()
        self.topk = topk  # top-k 候选数量
        self.num_classes = num_classes  # 总类别数
        self.bg_idx = num_classes  # 背景类别索引
        self.alpha = alpha  # 分类的权重参数
        self.beta = beta  # 定位的权重参数
        self.eps = eps  # 小值防止除零

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        计算任务对齐分配。

        参数：
            pd_scores (Tensor): 预测得分，形状为 (bs, num_total_anchors, num_classes)。
            pd_bboxes (Tensor): 预测边界框，形状为 (bs, num_total_anchors, 4)。
            anc_points (Tensor): 锚点的中心点坐标，形状为 (num_total_anchors, 2)。
            gt_labels (Tensor): 真实标签，形状为 (bs, n_max_boxes, 1)。
            gt_bboxes (Tensor): 真实边界框，形状为 (bs, n_max_boxes, 4)。
            mask_gt (Tensor): 掩码，用于指示哪些真实框有效，形状为 (bs, n_max_boxes, 1)。

        返回：
            target_labels (Tensor): 分配的目标标签，形状为 (bs, num_total_anchors)。
            target_bboxes (Tensor): 分配的目标边界框，形状为 (bs, num_total_anchors, 4)。
            target_scores (Tensor): 分配的目标分数，形状为 (bs, num_total_anchors, num_classes)。
            fg_mask (Tensor): 前景掩码，形状为 (bs, num_total_anchors)。
            target_gt_idx (Tensor): 分配的目标索引，形状为 (bs, num_total_anchors)。
        """
        self.bs = pd_scores.shape[0]  # 获取批量大小
        self.n_max_boxes = gt_bboxes.shape[1]  # 每张图片中最多的目标框数
        device = gt_bboxes.device  # 获取设备信息

        # 如果没有真实框，则直接返回默认值
        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),  # 所有锚点都分配为背景
                torch.zeros_like(pd_bboxes),  # 边界框全为零
                torch.zeros_like(pd_scores),  # 得分全为零
                torch.zeros_like(pd_scores[..., 0]),  # 前景掩码全为零
                torch.zeros_like(pd_scores[..., 0]),  # 目标索引全为零
            )

        try:
            # 调用具体的分配实现
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)  # 结果转回原设备

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        内部函数，具体实现任务对齐分配的逻辑。

        详细参数和返回值同 `forward` 方法。
        """
        # 获取正样本掩码、对齐度量和 IoU 重叠
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # 选择 IoU 最高的分配
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # 获取分配的目标标签、边界框和分数
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 归一化对齐度量
        align_metric *= mask_pos  # 仅保留正样本的对齐度量
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # 每个目标的最大对齐度量
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # 每个目标的最大 IoU
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric  # 调整目标分数

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        获取正样本的掩码 (b, max_num_obj, h*w)。

        参数:
            pd_scores: 预测的分数 (b, h*w, num_classes)。
            pd_bboxes: 预测的边界框 (b, h*w, 4)。
            gt_labels: 真实标签 (b, max_num_obj, 1)。
            gt_bboxes: 真实边界框 (b, max_num_obj, 4)。
            anc_points: 锚点坐标 (h*w, 2)。
            mask_gt: GT有效掩码 (b, max_num_obj, h*w)。

        返回:
            mask_pos: 最终正样本掩码。
            align_metric: 对齐度指标。
            overlaps: IoU矩阵。
        """
        # 获取锚点是否在GT中的掩码
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        
        # 计算对齐度指标和IoU，结合掩码 (mask_in_gts * mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        
        # 获取对齐度最高的 top-k 掩码
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        
        # 合并所有掩码，得到最终的正样本掩码
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        计算预测框和GT框之间的对齐度指标和IoU。

        参数:
            pd_scores: 预测分数 (b, h*w, num_classes)。
            pd_bboxes: 预测边界框 (b, h*w, 4)。
            gt_labels: GT类别标签 (b, max_num_obj, 1)。
            gt_bboxes: GT边界框 (b, max_num_obj, 4)。
            mask_gt: GT有效掩码 (b, max_num_obj, h*w)。

        返回:
            align_metric: 对齐度指标 (b, max_num_obj, h*w)。
            overlaps: IoU矩阵 (b, max_num_obj, h*w)。
        """
        na = pd_bboxes.shape[-2]  # 获取锚点数量 (h*w)
        mask_gt = mask_gt.bool()  # 确保掩码为布尔值
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device
        )  # 初始化IoU矩阵
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device
        )  # 初始化得分矩阵

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 索引数组
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # 批次索引
        ind[1] = gt_labels.squeeze(-1)  # 类别索引

        # 获取每个GT类别在每个锚点的预测得分
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        # 扩展预测框和GT框维度，并获取有效掩码对应的框
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        
        # 计算IoU并保存到对应位置
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # 计算对齐度指标
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        计算水平边界框的IoU。

        参数:
            gt_bboxes: GT边界框 (n, 4)。
            pd_bboxes: 预测边界框 (n, 4)。

        返回:
            IoU值，范围 [0, 1]。
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        根据给定指标选择 top-k 候选样本。

        参数:
            metrics: 指标矩阵 (b, max_num_obj, h*w)。
            largest: 是否选择最大的值 (True 表示选择最大值)。
            topk_mask: 可选的布尔掩码 (b, max_num_obj, topk)，用于过滤无效候选样本。

        返回:
            选择的 top-k 样本掩码 (b, max_num_obj, h*w)。
        """
        # 获取每个GT类别的 top-k 指标和对应索引
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        
        # 如果未提供掩码，则默认设置为 > eps 的值
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        
        # 将无效索引填充为 0
        topk_idxs.masked_fill_(~topk_mask, 0)

        # 统计每个候选样本在 top-k 中的出现次数
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        
        # 移除无效候选样本
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        计算正样本锚点的目标标签、目标边界框和目标分数。

        参数:
            gt_labels (Tensor): GT标签，形状为 (b, max_num_obj, 1)，b 为批次大小，max_num_obj 为最大目标数量。
            gt_bboxes (Tensor): GT边界框，形状为 (b, max_num_obj, 4)。
            target_gt_idx (Tensor): 分配给正样本锚点的 GT 索引，形状为 (b, h*w)，h*w 是锚点总数。
            fg_mask (Tensor): 正样本锚点掩码，布尔值，形状为 (b, h*w)。

        返回:
            Tuple[Tensor, Tensor, Tensor]:
                - target_labels: 正样本目标标签 (b, h*w)。
                - target_bboxes: 正样本目标边界框 (b, h*w, 4)。
                - target_scores: 正样本目标分数 (b, h*w, num_classes)。
        """
        # 计算批次索引，形状为 (b, 1)，用于与目标索引相加以定位每个批次的目标索引
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # 将索引平移到正确批次 (b, h*w)

        # 获取正样本目标标签，将 gt_labels 展平后索引对应的目标标签
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # 获取正样本目标边界框，使用平展后的 GT 边界框按索引提取
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]  # (b, h*w, 4)

        # 将目标标签的值限制在 [0, num_classes)
        target_labels.clamp_(0)

        # 初始化目标分数张量，形状为 (b, h*w, num_classes)，每个类别分数初始为 0
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, num_classes)

        # 为每个目标类别分配分数 1
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        # 根据前景掩码过滤无效的分数
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # 扩展前景掩码以匹配分数张量
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)  # 只保留正样本的分数

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        筛选在 GT 边界框内的正样本锚点。

        参数:
            xy_centers (Tensor): 锚点中心坐标，形状为 (h*w, 2)。
            gt_bboxes (Tensor): GT 边界框，形状为 (b, n_boxes, 4)。
            eps (float): 小数值，用于数值稳定性。默认值为 1e-9。

        返回:
            Tensor: 布尔掩码，表示正样本锚点，形状为 (b, n_boxes, h*w)。
        """
        n_anchors = xy_centers.shape[0]  # 锚点总数
        bs, n_boxes, _ = gt_bboxes.shape  # 获取批次大小和 GT 框数量

        # 将 GT 边界框拆分为左上角 (lt) 和右下角 (rb)
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # (b*n_boxes, 1, 2)

        # 计算锚点中心与 GT 边界框四个边的距离
        bbox_deltas = torch.cat(
            (xy_centers[None] - lt, rb - xy_centers[None]), dim=2
        ).view(bs, n_boxes, n_anchors, -1)  # 形状为 (b, n_boxes, h*w, 4)

        # 判断所有距离是否大于 eps，如果是，则锚点位于边界框内
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        当锚点分配给多个 GT 时，选择 IoU 最大的 GT。

        参数:
            mask_pos (Tensor): 正样本掩码，形状为 (b, n_max_boxes, h*w)。
            overlaps (Tensor): IoU 重叠矩阵，形状为 (b, n_max_boxes, h*w)。
            n_max_boxes (int): 最大 GT 框数量。

        返回:
            target_gt_idx (Tensor): 每个锚点分配的 GT 索引，形状为 (b, h*w)。
            fg_mask (Tensor): 前景掩码，形状为 (b, h*w)。
            mask_pos (Tensor): 更新后的正样本掩码，形状为 (b, n_max_boxes, h*w)。
        """
        # 按 GT 维度求和，转换为 (b, h*w)，表示每个锚点的正样本数量
        fg_mask = mask_pos.sum(-2)

        # 如果某些锚点分配给多个 GT，则只保留 IoU 最大的 GT
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # 多 GT 掩码
            max_overlaps_idx = overlaps.argmax(1)  # 找到 IoU 最大的 GT 索引 (b, h*w)

            # 初始化 IoU 最大的掩码
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            # 更新正样本掩码，只保留 IoU 最大的分配
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)  # 重新计算前景掩码

        # 找到每个锚点最终分配的 GT 索引
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        return target_gt_idx, fg_mask, mask_pos

class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """使用任务对齐度量为旋转边界框分配 Ground Truth 对象的分配器。"""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        计算旋转边界框的 IoU。

        参数:
            gt_bboxes: GT 边界框。
            pd_bboxes: 预测的边界框。

        返回:
            IoU 值。
        """
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)  # 调用 `probiou` 计算 IoU，限制范围在 [0, 1]

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        为旋转边界框选择包含正样本的锚点中心。

        参数:
            xy_centers (Tensor): 锚点中心，形状 (h*w, 2)。
            gt_bboxes (Tensor): GT 边界框，形状 (b, n_boxes, 5)。

        返回:
            Tensor: 布尔掩码，表示哪些锚点中心在边界框内，形状 (b, n_boxes, h*w)。
        """
        # 将旋转框从中心点 + 宽高 + 角度转换为顶点坐标 (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # 分割出四个顶点 (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a  # 计算向量 AB
        ad = d - a  # 计算向量 AD

        # 计算 AP (锚点中心到 A 点的向量)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)  # 向量 AB 的模长平方
        norm_ad = (ad * ad).sum(dim=-1)  # 向量 AD 的模长平方
        ap_dot_ab = (ap * ab).sum(dim=-1)  # AP 在 AB 方向的投影
        ap_dot_ad = (ap * ad).sum(dim=-1)  # AP 在 AD 方向的投影

        # 判断 AP 是否在矩形范围内
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    根据特征生成锚点。

    参数:
        feats: 特征图列表。
        strides: 每个特征图的步幅。
        grid_cell_offset: 网格单元偏移量，默认 0.5。

    返回:
        Tuple: 包含锚点坐标和步幅张量。
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        # 获取特征图的高度和宽度
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # 生成 x 和 y 方向上的偏移
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        # 创建网格
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        # 组合 x 和 y 坐标，存储为锚点
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # 创建步幅张量
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    将距离（ltrb）转换为边界框（xywh 或 xyxy）。

    参数:
        distance: 预测的 ltrb 值。
        anchor_points: 锚点中心。
        xywh: 是否返回中心点和宽高的格式。
        dim: 分割维度。

    返回:
        边界框张量。
    """
    lt, rb = distance.chunk(2, dim)  # 分割为左上角 (lt) 和右下角 (rb)
    x1y1 = anchor_points - lt  # 左上角坐标
    x2y2 = anchor_points + rb  # 右下角坐标
    if xywh:
        c_xy = (x1y1 + x2y2) / 2  # 计算中心点
        wh = x2y2 - x1y1  # 计算宽高
        return torch.cat((c_xy, wh), dim)  # 返回 xywh 格式
    return torch.cat((x1y1, x2y2), dim)  # 返回 xyxy 格式


def bbox2dist(anchor_points, bbox, reg_max):
    """
    将边界框（xyxy）转换为距离（ltrb）。

    参数:
        anchor_points: 锚点中心。
        bbox: 边界框。
        reg_max: 距离的最大值。

    返回:
        ltrb 格式的张量。
    """
    x1y1, x2y2 = bbox.chunk(2, -1)  # 分割为左上角和右下角
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # 限制范围


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    从预测的距离和角度解码旋转边界框。

    参数:
        pred_dist: 预测的距离，形状 (bs, h*w, 4)。
        pred_angle: 预测的角度，形状 (bs, h*w, 1)。
        anchor_points: 锚点中心，形状 (h*w, 2)。
        dim: 分割维度，默认为 -1。

    返回:
        旋转边界框张量，形状 (bs, h*w, 4)。
    """
    lt, rb = pred_dist.split(2, dim=dim)  # 分割为 lt 和 rb
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)  # 计算角度的余弦和正弦
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)  # 计算框的宽和高的一半
    x, y = xf * cos - yf * sin, xf * sin + yf * cos  # 计算旋转后的中心点偏移
    xy = torch.cat([x, y], dim=dim) + anchor_points  # 加上锚点中心
    return torch.cat([xy, lt + rb], dim=dim)  # 返回旋转边界框