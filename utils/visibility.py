from pathlib import Path

import numpy as np

from datasets.interop import pascal_texture_planes
from utils.misc import load_yaml_file


class VisibilityOracle:
    """
    VisibilityOracle object is responsible for determining which planar
      patches are visible (i.e. not self-occluded) for a certain 3D model
      given a viewpoint (i.e. spherical coords of the camera).
    """
    def __init__(self, pascal_class: str, visibility_dir: Path):
        self.pascal_class = pascal_class

        self.vis_dir = visibility_dir
        if not self.vis_dir.is_dir():
            raise FileNotFoundError(f'{self.vis_dir} not found.')

        self.vis_thresh = 0.15

        # This dictionary stores the visibilities for each cad model, for
        #  each class (i.e. 'car', 'chair' etc.). The dict values are loaded
        #  from disk in a lazy manner, when they happen to be needed.
        self.visibilities = {}

    def get_planes_visibility(self, cad_idx: str, az_deg: int, el_deg: int):
        """
        Get the list of planes which are visible given the current viewpoint

        Since loading and parsing a `.yaml` is very time consuming, the dict
          `self.visibilities` stores the content of files already loaded.
          For each example, a key is composed by pascal class and cad index.
          If the key is already in the visibility dict, its corresponding value
          is returned. Only in case of miss the file is loaded and parsed.
        """
        visible_planes = []

        key = f'{self.pascal_class}_cad_{cad_idx:03d}'

        try:
            visibilities = self.visibilities[key]
        except KeyError:
            visibility_file = self.vis_dir / f'pascal_{key}_visibility.yaml'
            self.visibilities[key] = load_yaml_file(visibility_file)
            visibilities = self.visibilities[key]

        # Round is FUNDAMENTAL here - @Luca why?
        az_deg = round(az_deg / visibilities['az_step']) * visibilities['az_step']
        el_deg = round(el_deg / visibilities['el_step']) * visibilities['el_step']

        el_deg = np.clip(el_deg, 0, 90)
        az_deg = np.clip(az_deg, 0, 360)

        for pl_name in pascal_texture_planes[self.pascal_class].keys():
            area = visibilities['areas'][el_deg][az_deg][pl_name][1]
            if area / visibilities['max_areas'][pl_name] > self.vis_thresh:
                visible_planes.append(pl_name)

        return visible_planes
