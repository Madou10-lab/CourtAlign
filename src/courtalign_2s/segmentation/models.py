from __future__ import annotations

import logging
import os
import os.path as osp
import time
from collections import defaultdict

import cv2
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.utils.functional import iou
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torch.utils.data import DataLoader

from courtalign_2s.segmentation import dataset_utils as du
from courtalign_2s.segmentation import model_utils as mu
from courtalign_2s.segmentation import utils

logger = logging.getLogger(__name__)


def canonical_architecture_name(model_name: str) -> str:
    key = model_name.replace("_", "").replace(" ", "").lower()
    if key in {"deeplabv3plus", "deeplabv3+"}:
        return "DeepLabV3Plus"
    raise ValueError(
        f"Unsupported CourtAlign-2S segmentation architecture: {model_name}. "
        "The released model uses DeepLabV3Plus."
    )


def create_courtalign_2s_segmentation_model(
    model_name: str,
    encoder: str,
    encoder_weights: str | None,
    n_classes: int,
    activation: str | None,
    output_stride: int,
) -> torch.nn.Module:
    """Instantiate the released DeepLabV3Plus and ResNet-34 segmentation model."""

    architecture = canonical_architecture_name(model_name)
    if encoder != "resnet34":
        raise ValueError(
            f"Unsupported CourtAlign-2S encoder: {encoder}. The released model uses resnet34."
        )
    common = {
        "encoder_name": encoder,
        "encoder_weights": encoder_weights,
        "classes": n_classes,
        "activation": activation,
    }
    if architecture != "DeepLabV3Plus":
        raise AssertionError(f"Unexpected architecture after validation: {architecture}")
    return smp.DeepLabV3Plus(encoder_output_stride=output_stride, **common)


class CourtAlign2SSegmentationModel:
    """Training wrapper for CourtAlign-2S segmentation models."""

    def __init__(
        self,
        dataset,
        experiment_id,
        experiment_name,
        model_name,
        input_height,
        input_width,
        encoder,
        encoder_weights,
        freeze_encoder,
        activation,
        optimizer,
        learning_rate,
        loss_function,
        batch_size,
        n_epochs,
        experiment_path,
        overlay_opacity,
        fine_tune,
        auto_batch_size=False,
        freeze_batch_norm=False,
    ):
        self.dataset = dataset
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.input_height = input_height
        self.input_width = input_width
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.freeze_encoder = freeze_encoder
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.experiment_path = experiment_path
        self.fine_tune = fine_tune
        self.overlay_opacity = overlay_opacity
        self.auto_batch_size = auto_batch_size
        self.freeze_batch_norm = freeze_batch_norm

        self.is_trained = False
        self.is_loaded = False
        self.saved_epoch = None

        os.makedirs(osp.join(experiment_path, "logs"), exist_ok=True)
        os.makedirs(osp.join(experiment_path, "checkpoints"), exist_ok=True)
        self.logfilehandler = logging.FileHandler(osp.join(experiment_path, "logs", "model_output.log"), "a")
        self.logfilehandler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
        logger.addHandler(self.logfilehandler)
        logger.info("-" * 40)
        logger.info(self.__class__.__name__ + " instance created")

    def prepare(self):
        self.prepare_model()
        self.freeze_model()
        self.get_batch_size()
        self.setup_loaders()
        self.setup_train()
        self.setup_progress()

    def prepare_model(self):
        self.model = torch.nn.Sequential()
        self.preprocessing_fn = lambda **kwargs: kwargs

    def get_batch_size(self):
        if not self.auto_batch_size:
            logger.info(f"Batch size fixed by config: {self.batch_size}")
            return
        batch_size_gpu = mu.get_batch_size(
            model=self.model,
            input_shape=(3, self.input_height, self.input_width),
            output_shape=(len(self.dataset), self.input_height, self.input_width),
            dataset_train_size=self.dataset.train_size(),
            dataset_valid_size=self.dataset.valid_size(),
        )
        logger.info(f"GPU batch size: {batch_size_gpu}")
        self.batch_size = int(min(self.batch_size, batch_size_gpu))
        logger.info(f"Batch size after gpu check: {self.batch_size}")

    def freeze_model(self):
        if self.freeze_encoder:
            n_train_param = 0
            for child in self.model.encoder.children():
                for param in child.parameters():
                    param.requires_grad = False
                    n_train_param += 1
            logger.info(f"Trainable parameters: {n_train_param}")
            logger.info(f"Non-trainable parameters: {self.get_model_nparameters() - n_train_param}")

    def setup_loaders(self):
        self.dataset.build_train(self.preprocessing_fn)
        self.train_loader = DataLoader(
            self.dataset.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.dataset.train_size() % self.batch_size == 1,
        )
        self.valid_loader = DataLoader(
            self.dataset.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.dataset.valid_size() % self.batch_size == 1,
        )
        self.test_loader = DataLoader(self.dataset.test_dataset, batch_size=1, shuffle=False)
        logger.info("Dataset loaders initialized")

    def setup_train(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = None
        self.metrics = []
        self.train_optimizer = None

    def train(self):
        self.training_time = 0
        try:
            self.gpu_usage = utils.get_gpu_memory()
        except Exception:
            self.gpu_usage = None
        logger.info(f"GPU usage: {self.gpu_usage}MB")
        self.is_trained = True

    def load_best_model(self, checkpoint_filename=None):
        checkpoint = "best_model.pth" if checkpoint_filename is None else checkpoint_filename
        checkpoint_path = osp.join(self.experiment_path, "checkpoints", checkpoint)
        if not osp.exists(checkpoint_path):
            logger.error(f"Can't find {checkpoint} checkpoint weights")
            return
        if not self.is_loaded or checkpoint_filename is not None:
            self.inference_model = torch.load(checkpoint_path, map_location=self.device)
            self.is_loaded = True

    def load_model(self, checkpoint_filename):
        self.inference_model = torch.load(
            osp.join(self.experiment_path, "checkpoints", checkpoint_filename),
            map_location=self.device,
        )

    def setup_progress(self):
        self.progression_output = osp.join(self.experiment_path, "Progression")
        utils.create_folder(self.progression_output)

        train_filenames = sorted(os.listdir(self.dataset.x_train_dir))
        if not train_filenames:
            logger.warning("No training images available for progression snapshots")
            self.progression_enabled = False
            return
        image_filename = osp.join(self.dataset.x_train_dir, train_filenames[0])
        mask_filename = osp.join(self.dataset.y_train_dir, train_filenames[0])

        self.image_progress = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)
        image_preprocess = self.preprocessing_fn(
            image=self.dataset.test_augmentation(image=self.image_progress)["image"]
        )["image"]
        self.image_preprocess_tensor = torch.from_numpy(image_preprocess).to(self.device).unsqueeze(0)
        gt_mask_progress = cv2.cvtColor(cv2.imread(mask_filename), cv2.COLOR_BGR2RGB)
        self.image_vis_progress = cv2.resize(self.image_progress, (self.input_width, self.input_height))
        self.gt_mask_vis_progress = du.one_hot_encode(gt_mask_progress, len(self.dataset))
        self.gt_mask_tensor_progress = torch.from_numpy(utils.to_tensor(self.gt_mask_vis_progress)).to(
            self.device
        ).unsqueeze(0)

        progression_columns = dict({"epoch": "int"}, **{c: "float" for c in self.dataset.class_names})
        self.progression_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in progression_columns.items()})
        self.progression_enabled = True

    def generate_progress(self, epoch):
        if not getattr(self, "progression_enabled", False):
            return
        pred_mask_tensor = self.model(self.image_preprocess_tensor)
        pred_mask = self.post_processing(pred_mask_tensor)
        pred_mask_overlay = utils.generate_overlay(
            pred_mask,
            self.image_vis_progress,
            self.overlay_opacity,
            self.dataset.colour_palette,
        )
        gt_mask_vis_overlay = utils.generate_overlay(
            utils.reverse_one_hot(self.gt_mask_vis_progress),
            self.image_vis_progress,
            self.overlay_opacity,
            self.dataset.colour_palette,
        )
        fp_mask_overlay = utils.generate_mask_fp(
            pred_mask,
            utils.reverse_one_hot(self.gt_mask_vis_progress),
            self.image_vis_progress,
            self.overlay_opacity,
            self.dataset.colour_palette[:],
        )
        final_image = utils.concat_images(
            [[self.image_vis_progress, gt_mask_vis_overlay], [pred_mask_overlay, fp_mask_overlay]]
        )
        cv2.imwrite(
            osp.join(self.progression_output, f"{epoch}.png"),
            cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR),
        )

        progression_partition = defaultdict(float)
        for i, c in enumerate(self.dataset.class_names):
            progression_partition[c] = iou(
                pred_mask_tensor,
                self.gt_mask_tensor_progress,
                ignore_channels=[n for n in range(self.dataset.n_classes) if n != i],
            ).detach().squeeze().cpu().numpy()
        row = dict({"epoch": epoch}, **progression_partition)
        self.progression_df = pd.concat([self.progression_df, pd.DataFrame([row])], ignore_index=True)
        self.progression_df.to_csv(osp.join(self.experiment_path, "iou_progression.csv"), index=False)

    def unload(self):
        self.is_loaded = False

    def post_processing(self, tensor):
        raise NotImplementedError

    def inference(self, image, checkpoint_filename=None):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def get_results(self, config):
        self.dataset.get_results(config)
        config["batch_size"] = self.batch_size
        config["n_parameters"] = self.get_model_nparameters()
        config["model_size"] = self.get_model_size()
        config["gpu_usage"] = self.gpu_usage
        config["training_time"] = self.training_time

    def get_model_size(self):
        param_size = sum(param.nelement() * param.element_size() for param in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
        return round((param_size + buffer_size) / 1024**2, 3)

    def get_model_nparameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def __del__(self):
        if hasattr(self, "logfilehandler"):
            logger.removeHandler(self.logfilehandler)


class DeepLabV3PlusModel(CourtAlign2SSegmentationModel):
    """DeepLabV3Plus segmentation model used by CourtAlign-2S."""

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        self.stride = min(mu.get_stride(self.input_height, self.input_width), 16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.fine_tune:
            self.model = torch.load(
                osp.join(self.experiment_path, "checkpoints", "best_model.pth"),
                map_location=self.device,
            )
        else:
            self.model = create_courtalign_2s_segmentation_model(
                model_name=self.model_name,
                encoder=self.encoder,
                encoder_weights=self.encoder_weights,
                n_classes=len(self.dataset),
                activation=self.activation,
                output_stride=self.stride,
            )
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        )
        logger.info("Model created")
        logger.info(f"Architecture: {canonical_architecture_name(self.model_name)}")
        logger.info(f"Output stride: {self.stride}")

    def setup_train(self):
        if self.loss_function != "DiceLoss":
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
        self.loss = DiceLoss()
        self.loss.name = self.loss_function
        if hasattr(self.loss, "_name"):
            self.loss._name = self.loss_function
        self.metrics = [IoU(threshold=0.5)]

        if self.optimizer == "adam":
            self.train_optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=self.learning_rate)])
        if self.optimizer == "sgd":
            self.train_optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        logger.info("Training initialized")

    def train(self):
        if self.is_trained and not self.fine_tune:
            logger.error("Model already trained")
            return
        try:
            train_epoch = mu.TrainEpoch(
                self.model,
                loss=self.loss,
                metrics=self.metrics,
                optimizer=self.train_optimizer,
                device=self.device,
                verbose=True,
                freeze_batch_norm=self.freeze_batch_norm,
            )
            valid_epoch = mu.ValidEpoch(
                self.model,
                loss=self.loss,
                metrics=self.metrics,
                device=self.device,
                verbose=True,
            )
            logger.info(f"Started training experiment {self.experiment_id}")
            self.train_miou = 0.0
            self.valid_miou = 0.0
            self.train_dice_loss = float("inf")
            self.valid_dice_loss = float("inf")
            self.train_logs_list, self.valid_logs_list = [], []
            self.train_logs_iter_list, self.valid_logs_iter_list = [], []
            start_time = time.time()
            try:
                for i in range(1, self.n_epochs + 1):
                    logger.info("")
                    logger.info(
                        f"Experiment: {self.experiment_id}_{self.experiment_name}. "
                        f"Model: {self.model_name}. Epoch: {i}"
                    )
                    train_logs, train_iter_logs = train_epoch.run(self.train_loader)
                    print(train_logs)
                    logger.info(
                        f"Train epoch {i} results: DiceLoss={train_logs['DiceLoss']:.5f}, "
                        f"miou={train_logs['iou_score']:.5f}"
                    )
                    valid_logs, valid_iter_logs = valid_epoch.run(self.valid_loader)
                    logger.info(
                        f"Valid epoch {i} results: DiceLoss={valid_logs['DiceLoss']:.5f}, "
                        f"miou={valid_logs['iou_score']:.5f}"
                    )
                    self.train_logs_list.append(train_logs)
                    self.valid_logs_list.append(valid_logs)
                    self.train_logs_iter_list.extend(train_iter_logs)
                    self.valid_logs_iter_list.extend(valid_iter_logs)

                    if self.valid_dice_loss >= valid_logs["DiceLoss"]:
                        self.valid_miou = valid_logs["iou_score"]
                        self.train_miou = train_logs["iou_score"]
                        self.valid_dice_loss = valid_logs["DiceLoss"]
                        self.train_dice_loss = train_logs["DiceLoss"]
                        torch.save(self.model, osp.join(self.experiment_path, "checkpoints", "best_model.pth"))
                        self.saved_epoch = i
                        logger.info("")
                        logger.info("Better results obtained and saved")

                    self.generate_progress(i)
                    if i > self.n_epochs - 12:
                        torch.save(self.model, osp.join(self.experiment_path, "checkpoints", f"best_model_{i}.pth"))
            except KeyboardInterrupt:
                logger.warning("Training interrupted by user")
            logger.info("")
            super().train()
            self.training_time = int(time.time() - start_time)
            logger.info(f"Training ended in {self.training_time / 3600} hours")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.exception("CUDA out of memory")
                raise RuntimeError("GPU memory is insufficient for this experiment") from exc
            raise

    def test(self):
        self.load_best_model()
        test_epoch = mu.ValidEpoch(
            self.inference_model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )
        logger.info("")
        logger.info("Evaluation on Test Data: ")
        test_logs, _ = test_epoch.run(self.test_loader)
        self.test_miou = test_logs["iou_score"]
        self.test_dice_loss = test_logs["DiceLoss"]
        logger.info(
            f"Test results: DiceLoss={test_logs['DiceLoss']:.5f}, "
            f"miou={test_logs['iou_score']:.5f}"
        )

    def post_processing(self, tensor):
        return utils.transpose_reverse_one_hot(tensor.detach().squeeze().cpu().numpy())

    def inference(self, image, preprocessing=True, checkpoint_filename=None):
        self.load_best_model(checkpoint_filename)
        if preprocessing:
            image = self.preprocessing_fn(image=self.dataset.test_augmentation(image=image)["image"])["image"]
        x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
        return self.inference_model(x_tensor)

    def get_results(self, config):
        super().get_results(config)
        config["stride"] = self.stride
        config["saved_epoch"] = self.saved_epoch
        config["train_miou"] = self.train_miou
        config["valid_miou"] = self.valid_miou
        config["train_dice_loss"] = self.train_dice_loss
        config["valid_dice_loss"] = self.valid_dice_loss
