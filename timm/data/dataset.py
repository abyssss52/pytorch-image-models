""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import numpy as np
import cv2
import torch
import tarfile
import logging
from PIL import Image

from .parsers import create_parser
_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    num = 0
    for root, subdirs, files in os.walk(folder, topdown=False):
        num += 1
        # print('进来的次数：%d'%num)
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        # print(label)
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
        # if num == 2:
        #     print(labels)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        # print(unique_labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


def load_class_map(filename, root=''):
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    else:
        assert False, 'Unsupported class map extension'
    return class_to_idx


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]



class Binocular_Dataset(data.Dataset):

    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index * 2]             # 取该路径下的彩色图片的地址
        img_ori = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')  # 读取图片
        if self.transform is not None:
            img_ori = self.transform(img_ori)              # 对图片进行transform
        img_ori = np.array(img_ori).astype(np.float32)     # Image 格式图片转换成array
        # if target is None:
        #     target = torch.zeros(1).long()
        path, target = self.samples[index * 2 + 1]         # 取该路径下的红外图片的地址
        img_ir = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')   # 读取图片
        if self.transform is not None:
            img_ir = self.transform(img_ir)                # 对图片进行transform
        img_ir = np.array(img_ir).astype(np.float32)       # Image 格式图片转换成array
        img_ir = img_ir[0, :, :]                           # 取第一个通道的数据
        img_ir = img_ir[np.newaxis, :]                     # 增加一个维度
        # print(img_ir)
        if target is None:
            target = torch.zeros(1).long()
        img = np.concatenate([img_ori,img_ir], axis=0)     # 原彩图与红外图叠加，组成4通道
        # print(img.shape)
        img = torch.from_numpy(img)                        # array转化成torch.tensor
        return img, target

    def __len__(self):
        return int(len(self.imgs)/2)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]




def _extract_tar_info(tarfile, class_to_idx=None, sort=True):
    files = []
    labels = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        label = os.path.basename(dirname)
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    tarinfo_and_targets = [(f, class_to_idx[l]) for f, l in zip(files, labels) if l in class_to_idx]
    if sort:
        tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets, class_to_idx


class DatasetTar(data.Dataset):

    def __init__(self, root, load_bytes=False, transform=None, class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isfile(root)
        self.root = root
        with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
            self.samples, self.class_to_idx = _extract_tar_info(tf, class_to_idx)
        self.imgs = self.samples
        self.tarfile = None  # lazy init in __getitem__
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.samples[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = iob.read() if self.load_bytes else Image.open(iob).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename

    def filenames(self, basename=False):
        fn = os.path.basename if basename else lambda x: x
        return [fn(x[0].name) for x in self.samples]



class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training, batch_size=batch_size)
        else:
            self.parser = parser
        self.transform = transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)



class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
