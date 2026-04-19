# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:07:39 2023

@author: PaRa
@author: igoforth
"""

import torch
import torch.nn.functional as F

FAIL_LOSS = 1e6


def l1_l2_loss(pred: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.5):
    l1 = F.l1_loss(pred, target, reduction="none")
    l2 = F.mse_loss(pred, target, reduction="none")
    total_loss = (l1_weight * l1 + (1 - l1_weight) * l2).mean()
    total_loss = torch.nan_to_num(
        total_loss, nan=0.0, posinf=FAIL_LOSS, neginf=-FAIL_LOSS
    )
    return total_loss


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weight: float = 10.0):
    mask = (target != 0).float()
    loss = (pred - target) ** 2
    weighted_loss = (loss * (1 + (weight - 1) * mask)).mean()
    weighted_loss = torch.nan_to_num(
        weighted_loss, nan=0.0, posinf=FAIL_LOSS, neginf=-FAIL_LOSS
    )
    return weighted_loss


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0):
    total_loss = F.huber_loss(pred, target, delta=delta, reduction="mean")
    total_loss = torch.nan_to_num(
        total_loss, nan=0.0, posinf=FAIL_LOSS, neginf=-FAIL_LOSS
    )
    return total_loss


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * bce_loss).mean()
    focal_loss = torch.nan_to_num(
        focal_loss, nan=0.0, posinf=FAIL_LOSS, neginf=-FAIL_LOSS
    )
    return focal_loss


def custom_doppler_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    peak_weight: float = 15.0,
    shift_weight: float = 10.0,
    spatial_weight: float = 8.0,
    background_weight: float = 5.0,
    image_size: int = 28,
) -> torch.Tensor:
    # Reshape predictions and targets to 2D
    pred_2d = pred.view(-1, image_size, image_size)
    target_2d = target.view(-1, image_size, image_size)

    # Basic MSE loss
    mse_loss = F.mse_loss(pred, target, reduction="mean")

    # Peak intensity loss
    peak_loss = F.mse_loss(pred.max(dim=-1)[0], target.max(dim=-1)[0])

    # Shift loss with spatial awareness
    pred_peak_pos = pred.argmax(dim=-1)
    target_peak_pos = target.argmax(dim=-1)
    pred_peak_y, pred_peak_x = pred_peak_pos // image_size, pred_peak_pos % image_size
    target_peak_y, target_peak_x = (
        target_peak_pos // image_size,
        target_peak_pos % image_size,
    )
    shift_loss = F.l1_loss(pred_peak_y.float(), target_peak_y.float()) + F.l1_loss(
        pred_peak_x.float(), target_peak_x.float()
    )

    # Spatial loss
    y_coords, x_coords = torch.meshgrid(
        torch.arange(image_size, device=pred.device),
        torch.arange(image_size, device=pred.device),
        indexing="ij",
    )
    coords = torch.stack((y_coords, x_coords), dim=-1).float()

    pred_weights = (
        F.softmax(pred_2d.view(-1, image_size * image_size), dim=-1)
        .view(-1, image_size, image_size)
        .unsqueeze(-1)
    )
    target_weights = (
        F.softmax(target_2d.view(-1, image_size * image_size), dim=-1)
        .view(-1, image_size, image_size)
        .unsqueeze(-1)
    )

    pred_center = (pred_weights * coords.unsqueeze(0)).sum(dim=(1, 2))
    target_center = (target_weights * coords.unsqueeze(0)).sum(dim=(1, 2))

    spatial_loss = F.mse_loss(pred_center, target_center)

    # Background suppression loss
    background_mask = target == 0
    background_loss = (pred * background_mask).mean()

    total_loss = (
        mse_loss
        + peak_weight * peak_loss
        + shift_weight * shift_loss
        + spatial_weight * spatial_loss
        + background_weight * background_loss
    )

    total_loss = torch.nan_to_num(
        total_loss, nan=0.0, posinf=FAIL_LOSS, neginf=-FAIL_LOSS
    )

    return total_loss


def gradual_custom_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    max_epochs: int,
    peak_weight: float = 5.0,
    shift_weight: float = 2.5,
) -> torch.Tensor:
    mse_loss = torch.mean((pred - target) ** 2)
    custom_loss = custom_doppler_loss(pred, target, peak_weight, shift_weight)

    # Gradually increase the weight of custom loss
    alpha = min(epoch / max_epochs, 1.0)
    total_loss = (1 - alpha) * mse_loss + alpha * custom_loss
    total_loss = torch.nan_to_num(
        total_loss, nan=0.0, posinf=FAIL_LOSS, neginf=-FAIL_LOSS
    )
    return total_loss
