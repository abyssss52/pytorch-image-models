import argparse
import torch
from torchvision import models

from timm.models import create_model


parser = argparse.ArgumentParser(description='pytorch CNN visualization by Grad-CAM')
parser.add_argument('--model_backbone', type=str, default='mnasnet_small', help='the type of model chosen.')  # mnasnet_small semnasnet_100  # mobilenetv2_100
parser.add_argument('--classes_num', type=int, default=2, help='the number of classification class.')
parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel.')
parser.add_argument('--input_size', type=int, default=224, help='the size of input.')
parser.add_argument('--torch_model', default='/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/MNASNet_small/checkpoint-51.pth.tar') # '/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/mobilenetv2_100_no_prefetcher/checkpoint-42.pth.tar'  # "/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/Live_Detection/model_best.pth.tar" #   # './checkpoints/train/20200319-182337-mobilenetv2_100-224/checkpoint-14.pth.tar'

args = parser.parse_args()

print("=====> load pytorch checkpoint...")
mymodel = create_model(
        model_name=args.model_backbone,
        pretrained=False,
        num_classes=args.classes_num,
        in_chans=args.input_channel,
        global_pool='avg')    # , checkpoint_path=args.torch_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
mymodel = mymodel.to(device)

# mymodel = models.resnet50(pretrained=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# mymodel = mymodel.to(device)

for name, module in mymodel._modules.items():
    print(name)
    print(module)
    print('=============================>')