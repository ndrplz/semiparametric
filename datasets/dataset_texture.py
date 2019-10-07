from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from datasets.dataset_stick import StickDataset
from datasets.interop import pascal_texture_planes
from utils.augmentation import MyRandomAffine
from utils.visibility import VisibilityOracle


def get_planes(image: np.ndarray, meta, pascal_class: str,
               vis_oracle: VisibilityOracle):
    src_kpoint_dict = meta['kpoints_2d']
    try:
        az = meta['vpoint'][0]
        el = meta['vpoint'][1]
    except KeyError:
        az = meta['azimuth']
        el = meta['elevation']

    h, w = image.shape[:2]

    visible_planes = vis_oracle.get_planes_visibility(meta['cad_idx'],
                                                      az_deg=int(az),
                                                      el_deg=int(el))

    planes = []
    kpoints_planes = []
    visibilities = []
    for pl_name in pascal_texture_planes[pascal_class].keys():
        pl_kp_names = pascal_texture_planes[pascal_class][pl_name]

        src_p2d = np.asarray(
            [list(map(float, src_kpoint_dict[k])) for k in pl_kp_names])
        src_p2d = np.int32(src_p2d * w)

        src_mask = cv2.fillPoly(np.zeros_like(image), [src_p2d],
                                color=(1, 1, 1))
        src_to_warp = image * src_mask

        planes.append(src_to_warp)
        kpoints_planes.append(src_p2d)
        visibilities.append(pl_name in visible_planes)

    return np.stack(planes, 0), kpoints_planes, np.stack(visibilities).astype(np.uint8)


def warp_unwarp_planes(src_planes: np.ndarray, src_planes_kpoints: List[np.ndarray],
                       dst_planes_kpoints: List[np.ndarray],
                       src_visibilities: np.ndarray, dst_visibilities: np.ndarray, pascal_class: str):
    planes_warped = np.zeros_like(src_planes, dtype=src_planes.dtype)
    planes_unwarped = np.zeros_like(src_planes, dtype=src_planes.dtype)

    keys = list(pascal_texture_planes[pascal_class].keys())
    symmetry_set = [keys.index('left'), keys.index('right')]

    for i, pl_name in enumerate(keys):
        """
        Conditions to skip:
        - pl not visible in src
        - pl not in symmetry and not visible in dst
        - pl in symmetry and neither one from the symmetry visible in dst

        """
        if not src_visibilities[i]:
            continue
        if i not in symmetry_set and not dst_visibilities[i]:
            continue
        if i in symmetry_set and 1 not in [dst_visibilities[j] for j in symmetry_set]:
            continue

        src_plane = src_planes[i]
        src_plane_kpoints = src_planes_kpoints[i]
        j = i
        if i in symmetry_set and not dst_visibilities[i]:
            j = symmetry_set[0] if i == symmetry_set[1] else symmetry_set[1]

        dst_plane_kpoints = dst_planes_kpoints[j]
        H12, _ = cv2.findHomography(src_plane_kpoints, dst_plane_kpoints)
        H21, _ = cv2.findHomography(dst_plane_kpoints, src_plane_kpoints)

        if H12 is not None and H21 is not None:
            h, w = src_planes[0].shape[0:2]
            src_warped = cv2.warpPerspective(src_plane, H12, dsize=(h, w))
            src_unwarped = cv2.warpPerspective(src_warped, H21, dsize=(h, w))

            planes_warped[j] = src_warped
            planes_unwarped[i] = src_unwarped

    return planes_warped, planes_unwarped


class TextureDatasetWithNormal(StickDataset):
    def __init__(self, dataset_dir: Path,
                 visibility_dir: Path,
                 ext: str = '*.png', resize_factor: float = 1.0,
                 demo_mode: bool = False, do_augmentation: bool = False,
                 use_LAB: bool = True, quantize_central: bool = False,
                 ):

        super(TextureDatasetWithNormal, self).__init__(dataset_dir, ext, resize_factor,
                                                       demo_mode, do_augmentation, use_LAB)

        self.quantize_central = quantize_central

        self.vis_oracle = VisibilityOracle(self.dataset_meta['pascal_class'],
                                           visibility_dir=visibility_dir)

    def __getitem__(self, idx):
        return self.prepare_example(image=self.data[self.mode + '_images'][idx],
                                    image_meta=self.data[self.mode + '_meta'][idx],
                                    image_normal=self.data[self.mode + '_normal'][idx])

    def _load_data(self):
        # Notice: Images are loaded only if normals are found
        train_names = sorted([p.name for p in self.folder.joinpath('normal_train').glob(self.ext)])
        test_names = sorted([p.name for p in self.folder.joinpath('normal_test').glob(self.ext)])

        train_normal_paths = sorted([self.folder.joinpath('normal_train', n) for n in train_names])
        test_normal_paths = sorted([self.folder.joinpath('normal_test', n) for n in test_names])

        train_image_paths = sorted([self.folder.joinpath('train', n) for n in train_names])
        test_image_paths = sorted([self.folder.joinpath('test', n) for n in test_names])

        train_meta_paths = sorted([self.folder.joinpath('meta_train', Path(n).stem + '.yaml') for n in train_names])
        test_meta_paths = sorted([self.folder.joinpath('meta_test', Path(n).stem + '.yaml') for n in test_names])

        if self.demo_mode:
            top_n = 100
            train_normal_paths = train_normal_paths[:top_n]
            test_normal_paths = test_normal_paths[:top_n]
            train_image_paths = train_image_paths[:top_n]
            test_image_paths = test_image_paths[:top_n]
            train_meta_paths = train_meta_paths[:top_n]
            test_meta_paths = test_meta_paths[:top_n]

        return {
            'train_normal': [self._preprocess(cv2.imread(str(f))) for f in train_normal_paths],
            'eval_normal': [self._preprocess(cv2.imread(str(f))) for f in test_normal_paths],
            'train_images': [self._preprocess(cv2.imread(str(f))) for f in train_image_paths],
            'eval_images': [self._preprocess(cv2.imread(str(f))) for f in test_image_paths],
            'train_meta': [self._load_metadata(f) for f in train_meta_paths],
            'eval_meta': [self._load_metadata(f) for f in test_meta_paths]
        }

    @staticmethod
    def quantization(img):
        Z = img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)
        return res2

    def prepare_example(self, image, image_meta, image_normal):

        src_image = image
        h, w = src_image.shape[:2]
        assert h == w
        offset = int(w * 0.1)
        pascal_class = self.dataset_meta['pascal_class']

        src_meta = image_meta
        src_normal = image_normal

        src_log_image = src_image.copy()
        src_central_crop = src_image[h // 2 - offset:h // 2 + offset, w // 2 - offset:w // 2 + offset].copy()
        src_central_crop = cv2.resize(src_central_crop, (w, h))
        if self.quantize_central:
            src_central_crop = self.quantization(src_central_crop)
            values, counts = np.unique(src_central_crop.reshape(-1, 3), return_counts=True, axis=0)
            src_central_crop = np.ones_like(src_central_crop) * values[np.argmax(counts)]

        dst_idx = np.random.randint(len(self))
        dst_image = self.data[self.mode + '_images'][dst_idx]
        dst_meta = self.data[self.mode + '_meta'][dst_idx]
        dst_normal = self.data[self.mode + '_normal'][dst_idx]

        dst_log_image = dst_image.copy()
        dst_central_crop = dst_image[h // 2 - offset:h // 2 + offset, w // 2 - offset:w // 2 + offset].copy()
        dst_central_crop = cv2.resize(dst_central_crop, (w, h))
        if self.quantize_central:
            dst_central_crop = self.quantization(dst_central_crop)
            values, counts = np.unique(dst_central_crop.reshape(-1, 3), return_counts=True, axis=0)
            dst_central_crop = np.ones_like(dst_central_crop) * values[np.argmax(counts)]

        src_pl_info = get_planes(src_image, src_meta, pascal_class, self.vis_oracle)
        src_planes, src_kpoints_planes, src_visibilities = src_pl_info

        dst_pl_info = get_planes(dst_image, dst_meta, pascal_class, self.vis_oracle)
        dst_planes, dst_kpoints_planes, dst_visibilities = dst_pl_info

        planes_warped, planes_unwarped = warp_unwarp_planes(src_planes, src_kpoints_planes, dst_kpoints_planes,
                                                            src_visibilities, dst_visibilities, pascal_class)

        # todo: two rows below are necessary for `augment_pascal.py` script
        # planes_warped, planes_unwarped = warp_unwarp_planes(dst_planes, dst_kpoints_planes, src_kpoints_planes,
        #                                                     dst_visibilities, src_visibilities, pascal_class)

        # Compute masks for both src and dst images
        bg_thresh = 20
        bg_color = 255
        src_image_masked = src_image.copy()
        src_bg_mask = np.all(src_normal <= bg_thresh, axis=-1)
        src_bg_mask = np.tile(src_bg_mask[..., np.newaxis], [1, 1, 3])
        src_image_masked[src_bg_mask] = bg_color

        dst_image_masked = dst_image.copy()
        dst_bg_mask = np.all(dst_normal <= bg_thresh, axis=-1)
        dst_bg_mask = np.tile(dst_bg_mask[..., np.newaxis], [1, 1, 3])
        dst_image_masked[dst_bg_mask] = bg_color

        src_fg_mask = np.bitwise_not(src_bg_mask).astype(np.uint8)
        dst_fg_mask = np.bitwise_not(dst_bg_mask).astype(np.uint8)

        # Data augmentation: all images and planes undergo the same warp
        if self.do_augmentation:
            # todo: notice that kpoints are not transformed
            affine = MyRandomAffine(degrees=10, translate=(0.1, 0.1),
                                    fillcolor=(0, 0, 0), shear=10,
                                    resample=Image.BICUBIC)
            affine.sample_params(w=w, h=h)

            src_planes = affine(*src_planes)
            planes_unwarped = affine(*planes_unwarped)
            src_normal = affine(src_normal)
            src_image = affine(src_image)
            src_image_masked = affine(src_image_masked,
                                      fillcolor=(255, 255, 255))
            src_fg_mask = affine(src_fg_mask)

        # Achtung! Make sure the same normalization is used for training
        planes_unwarped = self.planes_to_torch(planes_unwarped, to_LAB=self.use_LAB)
        planes_warped = self.planes_to_torch(planes_warped, to_LAB=self.use_LAB)
        src_planes = self.planes_to_torch(src_planes, to_LAB=self.use_LAB)

        src_image = self.to_torch(src_image, to_LAB=self.use_LAB)
        src_normal = self.to_torch(src_normal, to_LAB=self.use_LAB)

        src_log_image = self.to_torch(src_log_image, to_LAB=self.use_LAB)

        src_central_crop = self.to_torch(src_central_crop, to_LAB=self.use_LAB)
        dst_image = self.to_torch(dst_image, to_LAB=self.use_LAB)
        dst_normal = self.to_torch(dst_normal, to_LAB=self.use_LAB)

        src_image_masked = self.to_torch(src_image_masked, to_LAB=self.use_LAB)
        dst_image_masked = self.to_torch(dst_image_masked, to_LAB=self.use_LAB)
        dst_log_image = self.to_torch(dst_log_image, to_LAB=self.use_LAB)
        dst_central_crop = self.to_torch(dst_central_crop, to_LAB=self.use_LAB)

        # This is the CAD of src image - that is, the one that must be used to
        #  generate novel views of the source. However, the cad_idx is only
        #  available for Pascal3D+
        src_cad_idx = -1
        try:
            src_cad_idx = src_meta['cad_idx']
        except KeyError:
            pass
        if type(src_cad_idx) == str:
            src_cad_idx = -1  # todo: for shapenet cads

        # Pre-process foreground mask
        src_fg_mask = np.any(src_fg_mask, axis=-1, keepdims=True)
        src_fg_mask = torch.from_numpy(src_fg_mask.transpose(2, 0, 1)).float()

        return {
            'src_kpoints_planes': src_kpoints_planes,
            'dst_kpoints_planes': dst_kpoints_planes,
            'src_vs': src_visibilities,
            'dst_vs': dst_visibilities,
            'src_normal': src_normal,
            'dst_normal': dst_normal,
            'planes': src_planes,  # Texture planes
            'planes_warped': planes_warped,  # Texture planes
            'planes_unwarped': planes_unwarped,  # Texture planes
            'src_image': src_image,  # source image
            'dst_image': dst_image,  # Destination image
            'src_cad_idx': src_cad_idx,
            # todo: `image_meta` breaks mixed training but is still needed for competitors_predict.py
            # 'image_meta': src_meta,
            'src_image_masked': src_image_masked,
            'dst_image_masked': dst_image_masked,
            'dst_log_image': dst_log_image,
            'src_log_image': src_log_image,
            'src_central': src_central_crop,
            'dst_central': dst_central_crop,
            'src_fg_mask': src_fg_mask
        }

    def to_torch(self, x: np.ndarray, to_LAB: bool):
        """
        Convert a BGR image in 0..255 to a pytorch LAB Tensor in range -1..1
        """
        if to_LAB:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        x = Image.fromarray(x)
        return self.normalizer(ToTensor()(x))

    @staticmethod
    def planes_to_torch(planes, to_LAB: bool):
        planes = [p for p in planes]
        if to_LAB:
            planes = [cv2.cvtColor(p, cv2.COLOR_BGR2LAB) for p in planes]
        planes = np.stack(planes)
        planes = np.float32(planes) / 255.
        planes = np.transpose(planes, (0, 3, 1, 2))
        planes = (torch.from_numpy(planes) - 0.5) / 0.5
        return planes
