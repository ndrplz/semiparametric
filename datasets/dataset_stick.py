from PIL import Image
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from prettytable import PrettyTable
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

from utils.augmentation import MyRandomAffine


class StickDataset(Dataset):
    def __init__(self, folder: Path, ext: str='*.png', resize_factor: float=1.0,
                 demo_mode: bool=False, do_augmentation: bool=False, use_LAB: bool=True):
        self.do_augmentation = do_augmentation

        self.folder = Path(folder)
        if not self.folder.exists():
            raise OSError(f'Directory {self.folder} does not exist.')

        # Dataset-level metadata (e.g. pascal class, creation data etc.)
        self.dataset_meta = self._load_metadata(self.folder / 'dataset.yaml')

        self.resize_factor = resize_factor

        self.normalizer = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        self._mode = 'train'
        self.demo_mode = demo_mode
        self.use_LAB = use_LAB

        self.ext = ext
        assert self.ext in ['*.jpg', '*.png']

        print('Loading data... ', end='', flush=True)
        self.data = self._load_data()

        print('Done.')

    def prepare_example(self, image, image_stick, image_meta):
        if self.do_augmentation:
            h, w = image.shape[:2]
            affine = MyRandomAffine(degrees=10, translate=(0.1, 0.1),
                                    fillcolor=(0, 0, 0), shear=10,
                                    resample=Image.BICUBIC)
            affine.sample_params(w=w, h=h)
            image_stick = affine(image_stick)
            image = affine(image)

        # Convert all images to CIE-LAB color space
        image_input = image
        if self.use_LAB:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2LAB)
        image_target = image_input.copy()

        if self.use_LAB:
            image_stick = cv2.cvtColor(image_stick, cv2.COLOR_BGR2LAB)

        app_input = np.concatenate([image_input, image_stick], axis=-1)
        app_input = cv2.resize(app_input, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)

        app_input = np.float32(app_input) / 255.
        app_input = np.transpose(app_input, (2, 0, 1))
        app_input = self.normalizer(torch.from_numpy(app_input))

        image_log = self.normalizer(ToTensor()(Image.fromarray(image_input)))

        image_stick = Image.fromarray(image_stick)
        image_target = Image.fromarray(image_target)

        shape_input = self.normalizer(ToTensor()(image_stick))
        image_target = self.normalizer(ToTensor()(image_target))

        return {
            'app_input': app_input,        # Input for appearance branch
            'shape_input': shape_input,    # Input for shape branch
            'image_target': image_target,  # Target image
            'image_log': image_log,        # Input image for logging purposes
            'image_meta': image_meta       # Example metadata (e.g. CAD model)
        }

    def __getitem__(self, idx):
        return self.prepare_example(image=self.data[self.mode + '_images'][idx],
                                    image_stick=self.data[self.mode + '_sticks'][idx],
                                    image_meta=self.data[self.mode + '_meta'][idx])

    def __len__(self):
        return len(self.data[self.mode + '_images'])

    def __str__(self):
        mode = self.mode
        self.train()
        n_train = len(self)
        self.eval()
        n_eval = len(self)
        self.mode = mode

        table = PrettyTable(field_names=['train', 'eval'])
        table.add_row([n_train, n_eval])
        return str(table)

    def random_batch(self, filter_by_cad: [None, int], batch_size: int):
        """
        Load a batch of random examples from the current data split.

        If `filter_by_cad` is not None, sampled examples are filtered
          according to that CAD index.
        """
        idxs = np.arange(0, len(self), dtype=int)
        if filter_by_cad is not None:
            # Find the indexes of the given model in the current data split
            cad_idxs = [meta['cad_idx'] for meta in self.data[self.mode + '_meta']]
            idxs = np.argwhere(np.asarray(cad_idxs) == int(filter_by_cad)).squeeze()

        sampled_idxs = np.random.choice(idxs, size=min(len(idxs), batch_size), replace=False)

        batch = []
        for idx in sampled_idxs:
            example = self.prepare_example(image=self.data[self.mode + '_images'][idx],
                                           image_stick=self.data[self.mode + '_sticks'][idx],
                                           image_meta=self.data[self.mode + '_meta'][idx])
            batch.append(example)
        return batch

    def _load_data(self):
        # Images are loaded only if corresponding sticks are present
        train_stick_paths = sorted(list(self.folder.joinpath('stick_train').glob(self.ext)))
        test_stick_paths = sorted(list(self.folder.joinpath('stick_test').glob(self.ext)))

        train_image_paths = sorted([self.folder.joinpath('train', f.name) for f in train_stick_paths])
        test_image_paths = sorted([self.folder.joinpath('test', f.name) for f in test_stick_paths])

        train_meta_paths = sorted([self.folder.joinpath('meta_train', f.stem + '.yaml') for f in train_stick_paths])
        test_meta_paths = sorted([self.folder.joinpath('meta_test', f.stem + '.yaml') for f in test_stick_paths])

        if self.demo_mode:
            top_n = 100
            test_stick_paths = test_stick_paths[:top_n]
            train_stick_paths = train_stick_paths[:top_n]
            train_image_paths = train_image_paths[:top_n]
            test_image_paths = test_image_paths[:top_n]
            train_meta_paths = train_meta_paths[:top_n]
            test_meta_paths = test_meta_paths[:top_n]

        return {
            'train_images': [self._preprocess(cv2.imread(str(f))) for f in train_image_paths],
            'eval_images': [self._preprocess(cv2.imread(str(f))) for f in test_image_paths],
            'train_sticks': [self._preprocess(cv2.imread(str(f))) for f in train_stick_paths],
            'eval_sticks': [self._preprocess(cv2.imread(str(f))) for f in test_stick_paths],
            'train_meta': [self._load_metadata(f) for f in train_meta_paths],
            'eval_meta': [self._load_metadata(f) for f in test_meta_paths]
        }

    def _preprocess(self, image, interp=cv2.INTER_LINEAR):
        """ Possibly resize images when loading in RAM """
        image = cv2.resize(image, dsize=None, fx=self.resize_factor, fy=self.resize_factor, interpolation=interp)
        return image

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'eval']
        self._mode = value

    @staticmethod
    def _load_metadata(meta_file: Path):
        with meta_file.open(mode='r') as f:
            meta_info = yaml.load(f)
        return meta_info
