"""Boundary-aware segmentation loss for YOLO11-seg.

Adds an L1 term on Sobel edges of the predicted vs ground-truth masks to the
standard BCE mask loss, which sharpens thin crack boundaries.
"""
from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask


class BoundarySegmentationLoss(v8SegmentationLoss):
    def __init__(self, model, boundary_weight: float = 0.3, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.boundary_weight = boundary_weight

    @staticmethod
    def _edge(mask: torch.Tensor) -> torch.Tensor:
        dtype = mask.dtype
        device = mask.device
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        gx = F.conv2d(mask, sobel_x, padding=1)
        gy = F.conv2d(mask, sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, H, W)
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        bce = (crop_mask(bce, xyxy).mean(dim=(1, 2)) / area).sum()

        pred_prob = pred_mask.sigmoid()
        pred_edge = self._edge(pred_prob)
        gt_edge = self._edge(gt_mask)
        edge = (crop_mask((pred_edge - gt_edge).abs(), xyxy).mean(dim=(1, 2)) / area).sum()
        return bce + self.boundary_weight * edge


def enable_boundary_loss(model, boundary_weight: float = 0.3) -> None:
    """Monkey-patch a YOLO model instance so its next train() call uses boundary loss."""
    def init_criterion(self):
        self.criterion = BoundarySegmentationLoss(self, boundary_weight=boundary_weight)

    model.model.init_criterion = MethodType(init_criterion, model.model)
