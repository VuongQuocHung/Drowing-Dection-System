"""YOWOv2-tiny baseline using the leakage-free folds and scale augmentation."""

from copy import deepcopy
from importlib import import_module


config = deepcopy(import_module("config.YOWOv2_tiny").config)

config.update(
    {
        "fold": 10,
        "max_epoch": 15,
        "lr_epoch": [6, 10, 13],
        "zoom_out_prob": 0.30,
        "zoom_out_min_scale": 0.50,
        "zoom_out_fill": 114,
    }
)
