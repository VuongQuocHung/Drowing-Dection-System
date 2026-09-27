"""2Dseq-tiny settings aimed at unseen, wider pool scenes."""

from copy import deepcopy
from importlib import import_module


config = deepcopy(import_module("config.2Dseq_tiny").config)

config.update(
    {
        # The original windows overlap 46/48 sampled frames. A wider step
        # reduces near-duplicates and shortens each epoch considerably.
        "sequence_distance": 18,
        # Use the new video-group folds generated as 10..14.
        "fold": 10,
        "batch_size": 4,
        "num_workers": 2,
        # Give the grouped validation run enough time before LR decay.
        "max_epoch": 15,
        "lr_epoch": [6, 10, 13],
        # Synchronized for all frames/targets in a sequence. This exposes the
        # model to substantially smaller people without breaking motion.
        "zoom_out_prob": 0.30,
        "zoom_out_min_scale": 0.50,
        "zoom_out_fill": 114,
    }
)
