from __future__ import annotations

import logging
import os.path as osp
import time
from dataclasses import dataclass
from typing import Any

import torch
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

from courtalign_2s.segmentation import model_utils as mu
from courtalign_2s.segmentation.models import DeepLabV3PlusModel, CourtAlign2SSegmentationModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DicePhase:
    name: str
    start_epoch: int
    end_epoch: int
    active_class_names: tuple[str, ...] | None
    active_class_ids: tuple[int, ...] | None

    @property
    def uses_all_classes(self) -> bool:
        return self.active_class_ids is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "active_classes": "all" if self.uses_all_classes else list(self.active_class_names or ()),
            "active_class_ids": "all" if self.uses_all_classes else list(self.active_class_ids or ()),
        }


class ThreePhaseDiceSchedule:
    """Validated three-phase CourtAlign-2S badminton schedule."""

    def __init__(self, phases: tuple[DicePhase, ...], n_epochs: int):
        self.phases = phases
        self.n_epochs = n_epochs

    @classmethod
    def from_config(
        cls,
        phase_configs: list[dict[str, Any]],
        class_names: list[str],
        n_epochs: int,
    ) -> ThreePhaseDiceSchedule:
        if len(phase_configs) != 3:
            raise ValueError("ThreePhaseDiceLoss requires exactly three configured phases.")

        phases: list[DicePhase] = []
        covered_epochs: list[int] = []
        for index, phase_config in enumerate(phase_configs, start=1):
            expected_name = f"phase_{index}"
            name = str(phase_config.get("name", expected_name))
            if name != expected_name:
                raise ValueError(f"Expected phase name {expected_name!r}, got {name!r}.")

            start_epoch = int(phase_config["start_epoch"])
            end_epoch = int(phase_config["end_epoch"])
            if start_epoch < 1 or end_epoch < start_epoch:
                raise ValueError(f"Invalid epoch interval for {name}: {start_epoch}-{end_epoch}.")

            configured_classes = phase_config["active_classes"]
            if configured_classes == "all":
                active_class_names = None
                active_class_ids = None
            else:
                if not isinstance(configured_classes, list) or not configured_classes:
                    raise ValueError(f"{name}.active_classes must be 'all' or a non-empty list.")
                if len(set(configured_classes)) != len(configured_classes):
                    raise ValueError(f"{name}.active_classes contains duplicate class names.")
                missing = [class_name for class_name in configured_classes if class_name not in class_names]
                if missing:
                    raise ValueError(f"{name} references unknown classes: {missing}.")
                active_class_names = tuple(str(class_name) for class_name in configured_classes)
                active_class_ids = tuple(class_names.index(class_name) for class_name in active_class_names)

            phases.append(
                DicePhase(
                    name=name,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    active_class_names=active_class_names,
                    active_class_ids=active_class_ids,
                )
            )
            covered_epochs.extend(range(start_epoch, end_epoch + 1))

        if phases[0].active_class_ids is not None or phases[2].active_class_ids is not None:
            raise ValueError("ThreePhaseDiceLoss phases 1 and 3 must use all classes.")
        if phases[1].active_class_ids is None:
            raise ValueError("ThreePhaseDiceLoss phase 2 must focus on an explicit class subset.")
        if covered_epochs != list(range(1, n_epochs + 1)):
            raise ValueError(
                f"ThreePhaseDiceLoss phases must cover every epoch exactly once from 1 to {n_epochs}."
            )
        return cls(tuple(phases), n_epochs)

    def phase_for_epoch(self, epoch: int) -> DicePhase:
        if epoch < 1 or epoch > self.n_epochs:
            raise ValueError(f"Epoch {epoch} is outside the configured 1-{self.n_epochs} range.")
        for phase in self.phases:
            if phase.start_epoch <= epoch <= phase.end_epoch:
                return phase
        raise RuntimeError(f"No loss phase found for epoch {epoch}.")

    def as_dict(self) -> dict[str, Any]:
        return {"n_epochs": self.n_epochs, "phases": [phase.as_dict() for phase in self.phases]}


class ThreePhaseDiceDeepLabV3PlusModel(DeepLabV3PlusModel):
    """DeepLabV3+ training with the badminton three-phase Dice schedule.

    Phases 1 and 3 use ordinary SMP Dice over every channel. Phase 2 uses the same
    Dice implementation after ignoring all channels except the configured focus
    classes.
    """

    loss_log_name = "ThreePhaseDiceLoss"

    def __init__(
        self,
        dataset,
        loss_schedule: list[dict[str, Any]],
        checkpoint_epochs: list[int] | None = None,
        stop_after_epoch: int | None = None,
        **kwargs,
    ):
        self.dice_schedule = ThreePhaseDiceSchedule.from_config(
            loss_schedule,
            list(dataset.class_names),
            int(kwargs["n_epochs"]),
        )
        n_epochs = int(kwargs["n_epochs"])
        self.checkpoint_epochs = frozenset(int(epoch) for epoch in (checkpoint_epochs or []))
        if any(epoch < 1 or epoch > n_epochs for epoch in self.checkpoint_epochs):
            raise ValueError("checkpoint_epochs must fall within the configured training schedule.")
        self.stop_after_epoch = None if stop_after_epoch is None else int(stop_after_epoch)
        if self.stop_after_epoch is not None and not 1 <= self.stop_after_epoch <= n_epochs:
            raise ValueError("stop_after_epoch must fall within the configured training schedule.")
        if self.stop_after_epoch is not None and any(
            epoch > self.stop_after_epoch for epoch in self.checkpoint_epochs
        ):
            raise ValueError("A requested checkpoint occurs after stop_after_epoch.")
        super().__init__(dataset, **kwargs)

    @staticmethod
    def _named_dice_loss(ignore_channels: list[int] | None = None) -> DiceLoss:
        loss = DiceLoss(ignore_channels=ignore_channels)
        loss.name = ThreePhaseDiceDeepLabV3PlusModel.loss_log_name
        if hasattr(loss, "_name"):
            loss._name = ThreePhaseDiceDeepLabV3PlusModel.loss_log_name
        return loss

    def setup_train(self):
        self.loss_all_classes = self._named_dice_loss()
        focus_ids = self.dice_schedule.phases[1].active_class_ids
        if focus_ids is None:
            raise RuntimeError("ThreePhaseDiceLoss phase 2 has no focused classes.")
        ignored_ids = [class_id for class_id in range(self.dataset.n_classes) if class_id not in focus_ids]
        self.loss_focus_classes = self._named_dice_loss(ignore_channels=ignored_ids)

        # Held-out evaluation always uses the all-class objective.
        self.loss = self.loss_all_classes
        self.metrics = [IoU(threshold=0.5)]
        if self.optimizer == "adam":
            self.train_optimizer = torch.optim.Adam(
                [dict(params=self.model.parameters(), lr=self.learning_rate)]
            )
        elif self.optimizer == "sgd":
            self.train_optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.learning_rate,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        logger.info("Three-phase Dice training set up")
        for phase in self.dice_schedule.phases:
            logger.info("Loss schedule: %s", phase.as_dict())

    def _loss_for_phase(self, phase: DicePhase) -> DiceLoss:
        return self.loss_all_classes if phase.uses_all_classes else self.loss_focus_classes

    @staticmethod
    def _annotate_logs(logs: dict[str, Any], phase: DicePhase) -> dict[str, Any]:
        annotated = dict(logs)
        annotated["loss_phase"] = phase.name
        annotated["active_classes"] = (
            "all" if phase.uses_all_classes else list(phase.active_class_names or ())
        )
        annotated["checkpoint_selection_eligible"] = phase.uses_all_classes
        return annotated

    @staticmethod
    def _annotate_iteration_logs(
        logs: list[dict[str, Any]],
        phase: DicePhase,
    ) -> list[dict[str, Any]]:
        return [ThreePhaseDiceDeepLabV3PlusModel._annotate_logs(log, phase) for log in logs]

    def train(self):
        if self.is_trained and not self.fine_tune:
            logger.error("Model already trained")
            return

        phase_runners: dict[str, tuple[mu.TrainEpoch, mu.ValidEpoch]] = {}
        for phase in self.dice_schedule.phases:
            loss = self._loss_for_phase(phase)
            phase_runners[phase.name] = (
                mu.TrainEpoch(
                    self.model,
                    loss=loss,
                    metrics=self.metrics,
                    optimizer=self.train_optimizer,
                    device=self.device,
                    verbose=True,
                ),
                mu.ValidEpoch(
                    self.model,
                    loss=loss,
                    metrics=self.metrics,
                    device=self.device,
                    verbose=True,
                ),
            )

        logger.info("Started three-phase Dice experiment %s", self.experiment_id)
        self.train_miou = 0.0
        self.valid_miou = 0.0
        self.train_three_phase_dice = float("inf")
        self.valid_three_phase_dice = float("inf")
        self.train_logs_list, self.valid_logs_list = [], []
        self.train_logs_iter_list, self.valid_logs_iter_list = [], []
        start_time = time.time()

        try:
            for epoch in range(1, self.n_epochs + 1):
                phase = self.dice_schedule.phase_for_epoch(epoch)
                train_epoch, valid_epoch = phase_runners[phase.name]
                logger.info("")
                logger.info(
                    "Experiment: %s_%s. Model: %s. Epoch: %d. Loss phase: %s. Active classes: %s",
                    self.experiment_id,
                    self.experiment_name,
                    self.model_name,
                    epoch,
                    phase.name,
                    "all" if phase.uses_all_classes else list(phase.active_class_names or ()),
                )

                train_logs, train_iter_logs = train_epoch.run(self.train_loader)
                valid_logs, valid_iter_logs = valid_epoch.run(self.valid_loader)
                train_logs = self._annotate_logs(train_logs, phase)
                valid_logs = self._annotate_logs(valid_logs, phase)
                train_iter_logs = self._annotate_iteration_logs(train_iter_logs, phase)
                valid_iter_logs = self._annotate_iteration_logs(valid_iter_logs, phase)

                logger.info(
                    "Train epoch %d results: %s=%.5f, miou=%.5f",
                    epoch,
                    self.loss_log_name,
                    train_logs[self.loss_log_name],
                    train_logs["iou_score"],
                )
                logger.info(
                    "Valid epoch %d results: %s=%.5f, miou=%.5f",
                    epoch,
                    self.loss_log_name,
                    valid_logs[self.loss_log_name],
                    valid_logs["iou_score"],
                )
                self.train_logs_list.append(train_logs)
                self.valid_logs_list.append(valid_logs)
                self.train_logs_iter_list.extend(train_iter_logs)
                self.valid_logs_iter_list.extend(valid_iter_logs)

                # The focused phase has a different objective and is not comparable
                # numerically with the all-class validation loss.
                if phase.uses_all_classes and self.valid_three_phase_dice >= valid_logs[self.loss_log_name]:
                    self.valid_miou = valid_logs["iou_score"]
                    self.train_miou = train_logs["iou_score"]
                    self.valid_three_phase_dice = valid_logs[self.loss_log_name]
                    self.train_three_phase_dice = train_logs[self.loss_log_name]
                    torch.save(
                        self.model,
                        osp.join(self.experiment_path, "checkpoints", "best_model.pth"),
                    )
                    self.saved_epoch = epoch
                    logger.info("")
                    logger.info("Better comparable all-class validation result obtained and saved")

                self.generate_progress(epoch)
                if epoch in self.checkpoint_epochs:
                    torch.save(
                        self.model,
                        osp.join(self.experiment_path, "checkpoints", f"epoch_{epoch:03d}.pth"),
                    )
                    logger.info("Saved requested phase-boundary checkpoint at epoch %d", epoch)
                if epoch > self.n_epochs - 12:
                    torch.save(
                        self.model,
                        osp.join(self.experiment_path, "checkpoints", f"best_model_{epoch}.pth"),
                    )
                if self.stop_after_epoch is not None and epoch >= self.stop_after_epoch:
                    logger.info("Stopped controlled phase replay after epoch %d", epoch)
                    break
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")

        logger.info("")
        CourtAlign2SSegmentationModel.train(self)
        self.training_time = int(time.time() - start_time)
        logger.info("Training ended in %.4f hours", self.training_time / 3600)

    def test(self):
        self.load_best_model()
        test_epoch = mu.ValidEpoch(
            self.inference_model,
            loss=self.loss_all_classes,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )
        logger.info("")
        logger.info("Evaluation on test data with all-class Dice")
        test_logs, _ = test_epoch.run(self.test_loader)
        self.test_miou = test_logs["iou_score"]
        self.test_three_phase_dice = test_logs[self.loss_log_name]
        logger.info(
            "Test results: %s=%.5f, miou=%.5f",
            self.loss_log_name,
            self.test_three_phase_dice,
            self.test_miou,
        )

    def get_results(self, config):
        CourtAlign2SSegmentationModel.get_results(self, config)
        config["stride"] = self.stride
        config["saved_epoch"] = self.saved_epoch
        config["train_miou"] = self.train_miou
        config["valid_miou"] = self.valid_miou
        config["train_three_phase_dice"] = self.train_three_phase_dice
        config["valid_three_phase_dice"] = self.valid_three_phase_dice
        config["loss_schedule"] = self.dice_schedule.as_dict()
