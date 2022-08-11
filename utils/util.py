import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import random


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value,step, n=1):
        if self.writer is not None:
            if step % 200 == 0:
                self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_img):
        for t in self.transforms:
            img, anno_img = t(img, anno_img)
        return img, anno_img


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_img):
        width = img.size[0]
        height = img.size[1]

        scale = np.random.uniform(self.scale[0], self.scale[1])

        scale_w = int(scale * width)
        scale_h = int(scale * height)

        img = img.resize((scale_w, scale_h), Image.BICUBIC)
        anno_img = anno_img.resize((scale_w, scale_h), Image.NEAREST)

        if scale > 1.0:
            left = scale_w - width
            left = int(np.random.uniform(0, left))

            top = scale_h - height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left + width, top + height))
            anno_img = anno_img.crop((left, top, left + width, top + height))

        else:
            p_palette = anno_img.copy().getpalette()
            img_org = img.copy()
            anno_img_org = anno_img.copy()

            pad_width = width - scale_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height - scale_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))  # create new white input image the same as size with
            # input img
            img.paste(img_org, (pad_width_left, pad_height_top))

            anno_img = Image.new(anno_img.mode, (width, height), (0))  # create new white annotation image
            anno_img.paste(anno_img_org, (pad_width_left, pad_height_top))
            anno_img.putpalette(p_palette)

        return img, anno_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_img):
        rotate_angle = np.random.uniform(self.angle[0], self.angle[1])

        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_img = anno_img.rotate(rotate_angle, Image.NEAREST)
        return img, anno_img


class RandomMirror(object):
    def __call__(self, img, anno_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_img = ImageOps.mirror(anno_img)

        return img, anno_img


class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_img):
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        anno_img = anno_img.resize((self.input_size, self.input_size), Image.NEAREST)

        return img, anno_img


class NormalizeTensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_img):
        img = transforms.functional.to_tensor(img)

        img = transforms.functional.normalize(img, self.color_mean, self.color_std)
        anno_img = np.array(anno_img)
        #  white ambiguous (255)white -> 0(black)

        index = np.where(anno_img == 255)
        # print(index)
        anno_img[index] = 0

        anno_img = torch.from_numpy(anno_img)

        return img, anno_img


class RandomGaussianBlur(object):
    def __call__(self, img, anno_img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, anno_img


class DataTransform:
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation([-10, 10]),
                RandomMirror(),
                # RandomGaussianBlur(),
                Resize(input_size),
                NormalizeTensor(color_mean, color_std)
            ]),
            "val": Compose([
                Resize(input_size),
                NormalizeTensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_img):
        return self.data_transform[phase](img, anno_img)


def make_data_list(root_path, phase):
    # BASE_DIR = './VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'
    img_paths = []
    anno_paths = []
    file_names = root_path + '/ImageSets/Segmentation/' + phase + '.txt'
    with open(file_names, 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            img_path = './VOCdevkit/VOC2012/JPEGImages/' + line + '.jpg'
            img_paths.append(img_path)

            anno_path = './VOCdevkit/VOC2012/SegmentationClass/' + line + '.png'
            anno_paths.append(anno_path)

    return img_paths, anno_paths


def report_eta(steps, total):
    steps = steps % total
    speed = -1
    seconds = (total - steps) * speed
    hours = seconds // 3600
    minutes = (seconds - (hours * 3600)) // 60
    seconds = seconds % 60

    return hours, minutes, seconds


def decode_segmap(label_mask, dataset):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    return rgb


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])
