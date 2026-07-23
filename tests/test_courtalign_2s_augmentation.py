from pathlib import Path

import cv2
import numpy as np

from courtalign_2s.segmentation.dataset_utils import CourtSegmentationDataset, one_hot_encode
from courtalign_2s.segmentation.datasets import CourtAlign2SManifestDataset


class MaskAwareFlipAugmentation:
    def __init__(self):
        self.received_mask = None

    def __call__(self, *, image, mask):
        self.received_mask = mask.copy()
        return {"image": image, "mask": np.flip(mask, axis=1).copy()}


class ShapeCheckingAugmentation:
    def __init__(self):
        self.received_shape = None

    def __call__(self, *, image, mask):
        assert image.shape[:2] == mask.shape[:2]
        self.received_shape = (image.shape[:2], mask.shape[:2])
        return {"image": image, "mask": mask}


def test_court_segmentation_dataset_applies_augmentation_to_mask(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((2, 3, 3), dtype=np.uint8)
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    mask[0, 0] = [1, 1, 1]

    cv2.imwrite(str(images_dir / "sample.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(masks_dir / "sample.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    augmentation = MaskAwareFlipAugmentation()
    dataset = CourtSegmentationDataset(str(images_dir), str(masks_dir), n_classes=2, augmentation=augmentation)
    _, augmented_mask = dataset[0]

    assert augmentation.received_mask is not None
    assert augmentation.received_mask[0, 0, 1] == 1.0
    assert augmented_mask[0, 0, 1] == 0.0
    assert augmented_mask[0, 2, 1] == 1.0


def test_one_hot_encode_accepts_binary_255_foreground_masks():
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    mask[0, 1] = [255, 255, 255]
    mask[1, 2] = [255, 255, 255]

    encoded = one_hot_encode(mask, n_classes=2)

    assert encoded.shape == (2, 3, 2)
    assert encoded[0, 1, 1]
    assert encoded[1, 2, 1]
    assert encoded[0, 0, 0]


def test_one_hot_encode_preserves_class_id_triplet_masks():
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    mask[0, 1] = [1, 1, 1]
    mask[1, 2] = [2, 2, 2]

    encoded = one_hot_encode(mask, n_classes=3)

    assert encoded.shape == (2, 3, 3)
    assert encoded[0, 1, 1]
    assert encoded[1, 2, 2]
    assert encoded[0, 0, 0]


def test_court_segmentation_dataset_aligns_image_to_existing_mask_resolution(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    mask[0, 0] = [1, 1, 1]

    cv2.imwrite(str(images_dir / "sample.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(masks_dir / "sample.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    augmentation = ShapeCheckingAugmentation()
    dataset = CourtSegmentationDataset(str(images_dir), str(masks_dir), n_classes=2, augmentation=augmentation)
    augmented_image, augmented_mask = dataset[0]

    assert augmentation.received_shape == ((2, 3), (2, 3))
    assert augmented_image.shape == (2, 3, 3)
    assert augmented_mask.shape == (2, 3, 2)


def test_training_dataset_uses_train_augmentation(tmp_path: Path):
    dataset = object.__new__(CourtAlign2SManifestDataset)
    dataset.n_classes = 2
    train_aug = object()
    test_aug = object()
    dataset.train_augmentation = train_aug
    dataset.test_augmentation = test_aug

    for attr in (
        "x_train_dir",
        "y_train_dir",
        "x_valid_dir",
        "y_valid_dir",
        "x_test_dir",
        "y_test_dir",
    ):
        path = tmp_path / attr
        path.mkdir()
        setattr(dataset, attr, str(path))

    dataset.build_train(preprocessing_fn=None)

    assert dataset.train_dataset.augmentation is train_aug
    assert dataset.valid_dataset.augmentation is test_aug
    assert dataset.test_dataset.augmentation is test_aug
