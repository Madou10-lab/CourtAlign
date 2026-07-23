from __future__ import annotations

import logging
import os
import os.path as osp
import shutil
from pathlib import Path

import albumentations as album
import cv2

from courtalign_common.data.manifest import load_manifest
from courtalign_2s.segmentation import dataset_utils as du
from courtalign_2s.segmentation import utils
from courtalign_common.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)


DEFAULT_TENNIS_PALETTE = [[0, 0, 0], [255, 255, 255]]
DEFAULT_BADMINTON_PALETTE = [
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4],
    [5, 5, 5],
    [6, 6, 6],
    [7, 7, 7],
    [8, 8, 8],
    [9, 9, 9],
    [10, 10, 10],
    [11, 11, 11],
    [12, 12, 12],
    [13, 13, 13],
]


class CourtAlign2SManifestDataset:
    """CourtAlign-2S dataset backed by the frozen split manifests.

    Images and masks are staged under the run directory while retaining the
    released mask encoding and joint image-mask augmentation behavior.
    """

    def __init__(
        self,
        experiment_id: str,
        dataset_name: str,
        manifest_path: str | Path,
        experiment_path: str | Path,
        input_height: int,
        input_width: int,
        class_names: list[str],
        colour_palette: list[list[int]] | None = None,
        augmentation_colour_format: str = "hsv",
        augmentation_spatial: bool = True,
        augmentation_colour: bool = True,
    ) -> None:
        self.experiment_id = experiment_id
        self.dataset_name = dataset_name
        self.manifest_path = manifest_path
        self.experiment_path = str(resolve_repo_path(experiment_path))
        self.input_height = input_height
        self.input_width = input_width
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.class_ids = list(range(self.n_classes))
        self.back_ids: list[int] = []
        self.colour_palette = colour_palette or self._default_palette(dataset_name, self.n_classes)
        self.augmentation_colour_format = augmentation_colour_format
        self.augmentation_spatial = augmentation_spatial
        self.augmentation_colour = augmentation_colour

        log_dir = osp.join(self.experiment_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.logfilehandler = logging.FileHandler(osp.join(log_dir, "dataset_output.log"), "a")
        self.logfilehandler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
        logger.addHandler(self.logfilehandler)
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(self.__class__.__name__ + " instance created")
        logger.info(f"Class names: {self.class_names}")
        logger.info(f"Class ids: {self.class_ids}")
        logger.info(f"Class colour palette: {self.colour_palette}")

    @staticmethod
    def _default_palette(dataset_name: str, n_classes: int) -> list[list[int]]:
        if dataset_name == "tennis_fullcourt":
            return DEFAULT_TENNIS_PALETTE[:n_classes]
        if dataset_name == "badminton_zones":
            return DEFAULT_BADMINTON_PALETTE[:n_classes]
        return [[i, i, i] for i in range(n_classes)]

    def prepare(self) -> None:
        self.prepare_dataset()
        self.split_dataset()
        self.setup_augmentation()
        self.build_vis()

    def prepare_dataset(self) -> None:
        prepared_path = osp.join(self.experiment_path, "prepared_dataset")
        train_dir = osp.join(prepared_path, "train")
        valid_dir = osp.join(prepared_path, "valid")
        test_dir = osp.join(prepared_path, "test")

        self.x_train_dir = osp.join(train_dir, "source")
        self.y_train_dir = osp.join(train_dir, "mask")
        self.x_valid_dir = osp.join(valid_dir, "source")
        self.y_valid_dir = osp.join(valid_dir, "mask")
        self.x_test_dir = osp.join(test_dir, "source")
        self.y_test_dir = osp.join(test_dir, "mask")

        utils.create_folder(prepared_path)
        for path in [
            train_dir,
            valid_dir,
            test_dir,
            self.x_train_dir,
            self.y_train_dir,
            self.x_valid_dir,
            self.y_valid_dir,
            self.x_test_dir,
            self.y_test_dir,
        ]:
            os.makedirs(path, exist_ok=True)
        logger.info("Dataset output directories created")

    def split_dataset(self) -> None:
        split_to_dirs = {
            "train": (self.x_train_dir, self.y_train_dir),
            "val": (self.x_valid_dir, self.y_valid_dir),
            "valid": (self.x_valid_dir, self.y_valid_dir),
            "test": (self.x_test_dir, self.y_test_dir),
        }
        for row in load_manifest(self.manifest_path):
            image_dir, mask_dir = split_to_dirs[row.split]
            shutil.copy2(row.image_path, osp.join(image_dir, row.image_id))
            shutil.copy2(row.mask_path, osp.join(mask_dir, row.image_id))
        logger.info("Dataset copied from frozen manifest into train/valid/test folders")

    def setup_augmentation(self) -> None:
        train_transform = [album.Resize(height=self.input_height, width=self.input_width, always_apply=True)]
        if self.augmentation_colour_format == "gray":
            train_transform.append(album.ToGray(always_apply=True))
        if self.augmentation_colour_format == "hsv":
            train_transform.append(du.ToHSV(always_apply=True))
        if self.augmentation_colour_format == "bgr":
            train_transform.append(du.ToBGR(always_apply=True))
        if self.augmentation_spatial:
            train_transform.append(album.Perspective(scale=(0.05), always_apply=False, p=0.6))
            train_transform.append(
                album.ShiftScaleRotate(
                    rotate_limit=(-3, 3),
                    scale_limit=(0.05),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    shift_limit_x=(0, 0),
                    shift_limit_y=(0, 0),
                    always_apply=False,
                    p=0.6,
                )
            )
        if self.augmentation_colour:
            train_transform.append(album.ColorJitter(hue=0, always_apply=False, p=0.5))
            train_transform.append(album.GaussNoise(var_limit=10, p=0.1))

        test_transform = [album.Resize(height=self.input_height, width=self.input_width, always_apply=True)]
        if self.augmentation_colour_format == "gray":
            test_transform.append(album.ToGray(always_apply=True))
        if self.augmentation_colour_format == "hsv":
            test_transform.append(du.ToHSV(always_apply=True))
        if self.augmentation_colour_format == "bgr":
            test_transform.append(du.ToBGR(always_apply=True))

        self.train_augmentation = album.Compose(train_transform)
        self.test_augmentation = album.Compose(test_transform)

    def build_vis(self) -> None:
        self.train_dataset_vis = du.CourtSegmentationDataset(
            self.x_train_dir,
            self.y_train_dir,
            augmentation=self.train_augmentation,
            n_classes=self.n_classes,
        )
        self.valid_dataset_vis = du.CourtSegmentationDataset(
            self.x_valid_dir,
            self.y_valid_dir,
            augmentation=self.test_augmentation,
            n_classes=self.n_classes,
        )
        self.test_dataset_vis = du.CourtSegmentationDataset(
            self.x_test_dir,
            self.y_test_dir,
            augmentation=self.test_augmentation,
            n_classes=self.n_classes,
        )
        logger.info("Visual datasets built")

    def build_train(self, preprocessing_fn) -> None:
        self.train_dataset = du.CourtSegmentationDataset(
            self.x_train_dir,
            self.y_train_dir,
            augmentation=self.train_augmentation,
            preprocessing=preprocessing_fn,
            n_classes=self.n_classes,
        )
        self.valid_dataset = du.CourtSegmentationDataset(
            self.x_valid_dir,
            self.y_valid_dir,
            augmentation=self.test_augmentation,
            preprocessing=preprocessing_fn,
            n_classes=self.n_classes,
        )
        self.test_dataset = du.CourtSegmentationDataset(
            self.x_test_dir,
            self.y_test_dir,
            augmentation=self.test_augmentation,
            preprocessing=preprocessing_fn,
            n_classes=self.n_classes,
        )
        logger.info("Preprocessed datasets built")

    def train_size(self) -> int:
        return len(os.listdir(self.x_train_dir))

    def valid_size(self) -> int:
        return len(os.listdir(self.x_valid_dir))

    def test_size(self) -> int:
        return len(os.listdir(self.x_test_dir))

    def size(self) -> int:
        return self.train_size() + self.valid_size() + self.test_size()

    def __len__(self) -> int:
        return self.n_classes

    def get_results(self, config: dict) -> None:
        config["n_classes"] = self.n_classes
        config["train_size"] = self.train_size()
        config["valid_size"] = self.valid_size()
        config["test_size"] = self.test_size()

    def __del__(self):
        if hasattr(self, "logfilehandler"):
            logger.removeHandler(self.logfilehandler)
