import argparse
import os
import math
import numpy as np

from PIL import Image
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import transforms
from collections import OrderedDict

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Bincamera Validation')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='mobilenetv2_100',
                    help='model architecture (default: mobilenetv2_100)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=112, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=2,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/Live_Detection/checkpoint-6.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use half precision (fp16)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')

def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = 1

    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]

    return transforms.Compose(tfl)


def test(args):
    path = '/home/night/图片/temp/spoof/data/a_0000000001_0.png'
    img_ori = Image.open(path).convert('RGB')
    transform = transforms_imagenet_eval(img_size=112)
    img_ori = transform(img_ori)
    img_ori = np.array(img_ori).astype(np.float32)  # Image 格式图片转换成array

    path = '/home/night/图片/temp/spoof/data/a_0000000001_1.png'
    img_ir = Image.open(path).convert('RGB')
    img_ir = transform(img_ir)
    img_ir = np.array(img_ir).astype(np.float32)  # Image 格式图片转换成array

    img_ir = img_ir[0, :, :]
    img_ir = img_ir[np.newaxis, :]

    img = np.concatenate([img_ori, img_ir], axis=0)
    img = img[np.newaxis, :]
    img = torch.from_numpy(img)

    # img = np.ones((1,4,112,122)).astype(np.float32)
    # img = torch.from_numpy(img)

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=4,
        pretrained=False)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    logging.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model)

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        input = img.cuda()
        output = model(input)
        print(output)

if __name__ == '__main__':
    args = parser.parse_args()
    test(args)