from __future__ import annotations

import os
import shutil
import subprocess as sp

import cv2
import numpy as np


def get_gpu_memory() -> int:
    command = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
    memory_used_info = sp.check_output(command.split()).decode("ascii").split("\n")[1:-1]
    memory_used_value, _ = [int(x.split()[0]) for x in memory_used_info[0].split(",")]
    return memory_used_value


def create_folder(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def reverse_one_hot(image: np.ndarray) -> np.ndarray:
    return np.argmax(image, axis=-1)


def transpose_reverse_one_hot(image: np.ndarray) -> np.ndarray:
    return reverse_one_hot(np.transpose(image, (1, 2, 0)))


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def colour_code_segmentation(image: np.ndarray, label_values):
    colour_codes = np.array(label_values)
    return colour_codes[image.astype(int)]


def generate_overlay(mask, source, opacity, colour_palette):
    pred_mask_seg = colour_code_segmentation(mask, colour_palette)
    return cv2.addWeighted(source.astype(np.uint8), 1 - opacity, pred_mask_seg.astype(np.uint8), opacity, 0)


def generate_mask_fp(pred_mask, gt_mask, source, opacity, colour_palette):
    colour_palette_f = colour_palette[:]
    diff_mask = gt_mask == pred_mask
    color_seg = np.copy(gt_mask)
    color_seg[diff_mask] = len(colour_palette_f)
    colour_palette_f[0] = [45, 12, 230]
    colour_palette_f.append([0, 0, 0])
    pred_mask_seg = colour_code_segmentation(color_seg, colour_palette_f)
    return cv2.addWeighted(source.astype(np.uint8), 1 - opacity, pred_mask_seg.astype(np.uint8), opacity, 0)


def concat_images(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
