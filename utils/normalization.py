from typing import Union

import cv2
import numpy as np
import torch
from PIL.Image import Image
from torchvision.utils import make_grid


def to_tensor(image: Union[Image, np.ndarray], max_range: int=255):
    """
    Convert an image in range [0, `max_range`] to a torch tensor in range [-1, 1].

    :param image: np.ndarray or PIL image
    :param max_range: max value the image can assume
    :return: torch.FloatTensor
    """
    if isinstance(image, Image):
        image = np.asarray(image)

    image = np.float32(image)
    assert image.max() <= max_range

    image = image / max_range
    image = np.transpose(image, (2, 0, 1))
    image = image * 2. - 1.
    image = torch.from_numpy(image)
    return image


def to_image(x: Union[np.ndarray, torch.Tensor], from_LAB: bool):
    """
    Convert a tensor to a valid image for visualization.


    :param x: Input tensor is expected to be either LAB or BGR color space and to lie in range [-1, 1]
    :return x: Image BGR uint8 in range [0, 255]
    """
    assert len(x.shape) == 3, f'Unsupported image shape {x.shape}'

    try:
        x = x.to('cpu').detach().numpy()
        x = np.transpose(x, (1, 2, 0))
    except AttributeError:
        # Input tensor is already a numpy array
        pass

    x = (x + 1.) / 2 * 255
    x = np.clip(x, 0, 255)

    x = x.astype(np.uint8)
    if from_LAB:
        x = cv2.cvtColor(x, cv2.COLOR_LAB2BGR)
    return x


def planes_to_image(planes: torch.Tensor, from_LAB=False):
    """
    Given a planes tensor of shape (B, n_planes * 3, H, W) return a
     grid image in which each element of the grid is a set of planes
     represented as image.

    To cast a plane tensor (n_planes, H, W, 3) to image, planes are
     simply summed on first axis, assuming different planes have no
     overlap.
    """
    b_size, n_planes, h, w = planes.shape
    n_planes //= 3  # each plane has three color channels
    z = planes.view(b_size * n_planes, 3, h, w)
    z = [to_image(k, from_LAB) for k in z]
    z = np.asarray(z).reshape(b_size, n_planes, h, w, 3)
    z = np.clip(np.sum(z, 1), 0, 255).astype(np.uint8)
    z = torch.from_numpy(z.transpose((0, -1, 1, 2)))
    grid = make_grid(z)
    return grid.numpy().transpose(1, 2, 0)
