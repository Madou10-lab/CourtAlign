from __future__ import annotations

import os
import os.path as osp

import cv2
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform


def one_hot_encode(label: np.ndarray, n_classes: int) -> np.ndarray:
    """Decode CourtAlign-2S masks stored as class ids.

    Multi-class masks use RGB triplets such as (1, 1, 1). Binary masks may use
    either class identifiers or 0/255 encoding, so their foreground is decoded
    by occupancy instead of exact triplet equality.
    """
    if label.ndim == 2:
        class_ids = label
    elif label.ndim == 3 and np.array_equal(label[..., 0], label[..., 1]) and np.array_equal(label[..., 0], label[..., 2]):
        class_ids = label[..., 0]
    else:
        semantic_map = []
        for class_id in range(n_classes):
            colour = np.full((3), class_id)
            equality = np.equal(label, colour)
            semantic_map.append(np.all(equality, axis=-1))
        return np.stack(semantic_map, axis=-1)

    unique_values = np.unique(class_ids)
    if not np.all(np.isin(unique_values, np.arange(n_classes))):
        if n_classes != 2:
            raise ValueError(
                f"Unsupported mask values {unique_values.tolist()} for {n_classes} classes. "
                "Expected class-id masks."
            )
        class_ids = (class_ids > 0).astype(np.uint8)

    semantic_map = [(class_ids == class_id) for class_id in range(n_classes)]
    return np.stack(semantic_map, axis=-1)


def align_image_to_mask_shape(
    image: np.ndarray,
    mask: np.ndarray,
    image_path: str,
    mask_path: str,
) -> np.ndarray:
    """Match image resolution to a pre-generated mask before joint augmentation."""
    image_height, image_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    if (image_height, image_width) == (mask_height, mask_width):
        return image

    image_aspect = image_width / image_height
    mask_aspect = mask_width / mask_height
    if not np.isclose(image_aspect, mask_aspect, rtol=0.0, atol=1e-3):
        raise ValueError(
            "Image and mask have incompatible aspect ratios before augmentation: "
            f"{image_path} has shape {(image_height, image_width)}, "
            f"{mask_path} has shape {(mask_height, mask_width)}."
        )

    return cv2.resize(image, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)


class CourtSegmentationDataset(torch.utils.data.Dataset):
    """Directory-based dataset for CourtAlign-2S segmentation.

    Augmentations are applied jointly to image and mask so stochastic spatial
    transforms keep the segmentation labels aligned with the transformed frame.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        n_classes: int = 2,
        augmentation=None,
        preprocessing=None,
    ) -> None:
        self.image_paths = [osp.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [osp.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i: int):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.n_classes).astype("float")
        image = align_image_to_mask_shape(image, mask, self.image_paths[i], self.mask_paths[i])

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def get_image_filename(self, i: int) -> str:
        return osp.basename(self.image_paths[i])

    def __len__(self) -> int:
        return len(self.image_paths)


class ToHSV(ImageOnlyTransform):
    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def get_transform_init_args_names(self):
        return ()


class ToBGR(ImageOnlyTransform):
    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def get_transform_init_args_names(self):
        return ()
