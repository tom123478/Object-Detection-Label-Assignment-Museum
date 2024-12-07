import torch
import numpy as np
import math

def xywhr2xyxyxyxy(x):
    """
    将批量的旋转边界框 (OBB) 从 [中心点 (x, y), 宽 (w), 高 (h), 旋转角度] 格式
    转换为 [顶点1 (x1, y1), 顶点2 (x2, y2), 顶点3 (x3, y3), 顶点4 (x4, y4)] 格式。

    参数:
        x (numpy.ndarray | torch.Tensor): 边界框输入，格式为 [cx, cy, w, h, rotation]，形状 (n, 5) 或 (b, n, 5)。

    返回:
        numpy.ndarray | torch.Tensor: 转换后的边界框顶点，形状 (n, 4, 2) 或 (b, n, 4, 2)。
    """
    # 根据输入类型选择使用 PyTorch 或 NumPy 的相关函数
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    # 提取中心点 (cx, cy)、宽、高和旋转角度
    ctr = x[..., :2]  # 中心点坐标
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))  # 宽、高、角度

    # 计算旋转角度的余弦和正弦值
    cos_value, sin_value = cos(angle), sin(angle)

    # 计算两个方向上的向量
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]  # 沿宽方向的向量
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]  # 沿高方向的向量
    vec1 = cat(vec1, -1)  # 合并 vec1 的 x 和 y 方向
    vec2 = cat(vec2, -1)  # 合并 vec2 的 x 和 y 方向

    # 计算四个顶点的坐标
    pt1 = ctr + vec1 + vec2  # 顶点1
    pt2 = ctr + vec1 - vec2  # 顶点2
    pt3 = ctr - vec1 - vec2  # 顶点3
    pt4 = ctr - vec1 + vec2  # 顶点4

    # 返回四个顶点的坐标
    return stack([pt1, pt2, pt3, pt4], -2)


def _get_covariance_matrix(boxes):
    """
    根据 OBB（旋转边界框）生成协方差矩阵。

    参数:
        boxes (torch.Tensor): 旋转边界框张量，形状为 (N, 5)，格式为 [x, y, w, h, rotation]。

    返回:
        torch.Tensor: 原始边界框对应的协方差矩阵。
    """
    # 计算高斯边界框（忽略中心点，只计算宽高和旋转角度的相关信息）
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)  # 标准差平方
    a, b, c = gbbs.split(1, dim=-1)  # 分割宽、高和旋转角度
    cos = c.cos()  # 旋转角度的余弦
    sin = c.sin()  # 旋转角度的正弦
    cos2 = cos.pow(2)  # 余弦平方
    sin2 = sin.pow(2)  # 正弦平方

    # 返回协方差矩阵的组成部分
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    计算两个旋转边界框之间的概率 IoU。

    参数:
        obb1 (torch.Tensor): GT 边界框，形状为 (N, 5)，格式为 xywhr。
        obb2 (torch.Tensor): 预测边界框，形状为 (N, 5)，格式为 xywhr。
        CIoU (bool, optional): 是否计算 CIoU。默认为 False。
        eps (float, optional): 小值，用于避免除以零。默认为 1e-7。

    返回:
        torch.Tensor: OBB 的相似性分数，形状为 (N,)。
    """
    # 提取中心点坐标
    x1, y1 = obb1[..., :2].split(1, dim=-1)  # obb1 的中心点
    x2, y2 = obb2[..., :2].split(1, dim=-1)  # obb2 的中心点

    # 获取协方差矩阵
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    # 计算概率 IoU 的三个部分
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        (
            ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
            / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        )
        + eps
    ).log() * 0.5

    # 计算 Bhattacharyya 距离和 IoU
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()  # Bhattacharyya 距离
    iou = 1 - hd  # 转化为 IoU

    # 如果需要计算 CIoU，则加入形状一致性损失项
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)  # obb1 的宽和高
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)  # obb2 的宽和高
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)  # 形状一致性损失
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))  # 加权因子
        return iou - v * alpha  # CIoU
    return iou  # 返回 IoU



def batch_probiou(obb1, obb2, eps=1e-7):
    """
    批量计算旋转边界框之间的概率 IoU。

    参数:
        obb1 (torch.Tensor | np.ndarray): GT 边界框，形状为 (N, 5)，格式为 xywhr。
        obb2 (torch.Tensor | np.ndarray): 预测边界框，形状为 (M, 5)，格式为 xywhr。
        eps (float, optional): 小值，用于避免除以零。默认为 1e-7。

    返回:
        torch.Tensor: 相似性矩阵，形状为 (N, M)。
    """
    # 如果输入为 NumPy 数组，转换为 PyTorch 张量
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    # 提取中心点坐标
    x1, y1 = obb1[..., :2].split(1, dim=-1)  # obb1 的中心点
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))  # obb2 的中心点 (扩展维度以匹配批次)

    # 获取协方差矩阵
    a1, b1, c1 = _get_covariance_matrix(obb1)  # obb1 的协方差矩阵
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))  # obb2 的协方差矩阵 (扩展维度)

    # 计算概率 IoU 的三个部分
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        (
            ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
            / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        )
        + eps
    ).log() * 0.5

    # 计算 Bhattacharyya 距离和 IoU
    bd = (t1 + t2 + t3).clamp(eps, 100.0)  # Bhattacharyya 距离
    hd = (1.0 - (-bd).exp() + eps).sqrt()  # Bhattacharyya 距离转为 Hellinger 距离
    return 1 - hd  # 返回 IoU

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算两个边界框之间的 IoU 或扩展的 IoU（GIoU、DIoU 或 CIoU）。

    参数:
        box1 (torch.Tensor): 边界框1，形状为 (1, 4)。
        box2 (torch.Tensor): 边界框2，形状为 (n, 4)。
        xywh (bool): 如果为 True，输入格式为 (x, y, w, h)，否则为 (x1, y1, x2, y2)。
        GIoU (bool): 如果为 True，计算 Generalized IoU。
        DIoU (bool): 如果为 True，计算 Distance IoU。
        CIoU (bool): 如果为 True，计算 Complete IoU。
        eps (float): 小值，用于避免除以零。

    返回:
        torch.Tensor: IoU 或扩展的 IoU 值。
    """
    # 解析输入边界框的坐标
    if xywh:  # 如果输入格式为 (x, y, w, h)，转换为 (x1, y1, x2, y2)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_  # box1 左上角和右下角 x 坐标
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_  # box1 左上角和右下角 y 坐标
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_  # box2 左上角和右下角 x 坐标
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_  # box2 左上角和右下角 y 坐标
    else:  # 如果输入格式为 (x1, y1, x2, y2)，直接解析坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps  # 计算宽和高
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 计算交集面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # 计算并集面积
    union = w1 * h1 + w2 * h2 - inter + eps

    # 计算 IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:  # 如果需要计算扩展的 IoU
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 包围矩形的宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 包围矩形的高度
        if CIoU or DIoU:  # 如果需要计算 DIoU 或 CIoU
            c2 = cw.pow(2) + ch.pow(2) + eps  # 包围矩形对角线平方
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # 中心点距离平方
            if CIoU:  # 计算 CIoU
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)  # 形状一致性损失
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))  # 权重因子
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # 包围矩形的面积
        return iou - (c_area - union) / c_area  # GIoU
    return iou  # 返回 IoU

