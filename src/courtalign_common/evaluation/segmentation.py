from __future__ import annotations

import numpy as np


def confusion_matrix_from_masks(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    valid = (y_true >= 0) & (y_true < n_classes)
    encoded = n_classes * y_true[valid].astype(np.int64) + y_pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def metrics_from_confusion(confusion: np.ndarray, class_names: list[str]) -> dict:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1)

    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou = np.divide(tp, denom_iou, out=np.full_like(tp, np.nan), where=denom_iou > 0)
    dice = np.divide(2 * tp, denom_dice, out=np.full_like(tp, np.nan), where=denom_dice > 0)
    pixel_accuracy = tp.sum() / confusion.sum() if confusion.sum() else np.nan

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": name,
                "iou": float(iou[idx]) if not np.isnan(iou[idx]) else None,
                "dice": float(dice[idx]) if not np.isnan(dice[idx]) else None,
                "support_pixels": int(support[idx]),
            }
        )

    return {
        "mean_iou": float(np.nanmean(iou)),
        "mean_dice": float(np.nanmean(dice)),
        "pixel_accuracy": float(pixel_accuracy),
        "per_class": per_class,
    }
