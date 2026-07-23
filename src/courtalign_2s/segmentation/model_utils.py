from __future__ import annotations

from time import sleep
import sys
import typing as t

import albumentations as album
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from tqdm import tqdm


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    transforms = []
    if preprocessing_fn:
        transforms.append(album.Lambda(image=preprocessing_fn))
    transforms.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(transforms)


def get_stride(height: int, width: int) -> int:
    stride = 2
    while height % stride == 0 and width % stride == 0 and stride < 64:
        stride *= 2
    return stride // 2


def get_batch_size(
    model: nn.Module,
    input_shape: t.Tuple[int, int, int],
    output_shape: t.Tuple[int, int, int],
    dataset_train_size: int,
    dataset_valid_size: int,
    max_batch_size: int = 16,
    num_iterations: int = 5,
) -> int:
    max_batch_size = min(max_batch_size, dataset_train_size // 2, dataset_valid_size)
    device = torch.device("cuda")
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())
    inputs, targets, loss = None, None, None

    batch_size = 2
    while True:
        if batch_size > max_batch_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
            sleep(3)
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer, inputs, targets, loss, device
    torch.cuda.empty_cache()
    return batch_size


def get_loss_name(loss) -> str:
    public_name = getattr(loss, "name", None)
    if public_name:
        return str(public_name)
    private_name = getattr(loss, "_name", None)
    if private_name:
        return str(private_name)
    return loss.__class__.__name__


def freeze_batch_norm_layers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


class TrainEpoch(smp.utils.train.TrainEpoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True, freeze_batch_norm=False):
        super().__init__(model=model, loss=loss, metrics=metrics, device=device, verbose=verbose, optimizer=optimizer)
        self.freeze_batch_norm = freeze_batch_norm

    def run(self, dataloader):
        self.on_epoch_start()
        if self.freeze_batch_norm:
            freeze_batch_norm_layers(self.model)
        logs_iterations = []
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        loss_name = get_loss_name(self.loss)

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {loss_name: loss_meter.mean}
                logs.update(loss_logs)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs_iterations.append(metrics_logs | loss_logs)
                logs.update(metrics_logs)

                if self.verbose:
                    iterator.set_postfix_str(self._format_logs(logs))
        return logs, logs_iterations


class ValidEpoch(smp.utils.train.ValidEpoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(model=model, loss=loss, metrics=metrics, device=device, verbose=verbose)

    def run(self, dataloader):
        self.on_epoch_start()
        logs_iterations = []
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        loss_name = get_loss_name(self.loss)

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {loss_name: loss_meter.mean}
                logs.update(loss_logs)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs_iterations.append(metrics_logs | loss_logs)
                logs.update(metrics_logs)

                if self.verbose:
                    iterator.set_postfix_str(self._format_logs(logs))
        return logs, logs_iterations
