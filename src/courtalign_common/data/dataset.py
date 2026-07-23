from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from courtalign_common.data.manifest import ManifestRow


def to_hsv_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def load_image_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def one_hot(mask: np.ndarray, n_classes: int) -> np.ndarray:
    encoded = np.zeros((n_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for class_id in range(n_classes):
        encoded[class_id] = (mask == class_id).astype(np.float32)
    return encoded


class SegmentationDataset:
    def __init__(
        self,
        rows: list[ManifestRow],
        n_classes: int,
        color_space: str = "rgb",
        resize: tuple[int, int] | None = None,
        return_one_hot: bool = True,
    ):
        self.rows = rows
        self.n_classes = n_classes
        self.color_space = color_space
        self.resize = resize
        self.return_one_hot = return_one_hot

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = load_image_rgb(row.image_path)
        mask = load_mask(row.mask_path)

        if self.resize is not None:
            height, width = self.resize
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        if self.color_space == "hsv":
            image = to_hsv_rgb_image(image)

        image_tensor = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
        if self.return_one_hot:
            mask_tensor = one_hot(mask, self.n_classes)
        else:
            mask_tensor = mask.astype(np.int64)

        return image_tensor, mask_tensor, row.image_id
