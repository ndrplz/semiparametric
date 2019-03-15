"""
 Miscellaneous functions that may be useful for more than one Dataset
"""
import numpy as np
import torch


def mask_to_torch(mask: np.ndarray):
    """
    Convert a three-channel binary mask from np.uint8 channel last
     to a single-channel binary mask torch.float channel first
    """
    # Sanity-check that mask comes as expected
    assert not mask.min() < 0
    assert not mask.max() > 1
    assert len(np.unique(mask)) in [1, 2]
    assert len(mask.shape) == 3
    assert mask.shape[-1] == 3

    mask = np.float32(mask).transpose(2, 0, 1)
    return torch.from_numpy(mask[:1])


def seg_to_image(seg: np.ndarray, color_dict: dict):
    """
    Convert part segmentation to uint8 BGR image in range 0-255
    """
    assert len(seg.shape) == 2

    part_keys = sorted(color_dict.keys())  # deterministic order

    img_out = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.float32)
    for i, part in enumerate(part_keys):
        color = color_dict[part]
        class_idx = i + 1   # plus one for BG
        indexes = np.argwhere(seg == class_idx)
        if len(indexes) == 0:
            continue
        img_out[indexes[:, 0], indexes[:, 1]] = color
    img_out = np.uint8(img_out * 255)[:, :, ::-1]
    return img_out
