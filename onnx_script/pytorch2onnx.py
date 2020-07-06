import torch
import torchvision
import torch.onnx
from torchvision import models
from torch.autograd import Variable

from timm.models import create_model

import argparse
import onnx
import onnxsim


# An instance of your model
# mymodel = models.mobilenet_v2(pretrained=True)
# # mymodel.load_state_dict(torch.load('mobilenet_v2.pth.tar'))


parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--model_backbone',type=str, default='mnasnet_small',help='the type of model chosen.')     # semnasnet_100  mobilenetv2_100  mnasnet_small
parser.add_argument('--classes_num', type=int, default=2, help='the number of classification class.')
parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel.')
parser.add_argument('--input_size', type=int, default=224, help='the size of input.')
parser.add_argument('--torch_model', default='/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/MNASNet_small/checkpoint-51.pth.tar') # '/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/mobilenetv2_100_no_prefetcher/checkpoint-63_new.pth.tar'  # "/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/Live_Detection/model_best.pth.tar" #   # './checkpoints/train/20200319-182337-mobilenetv2_100-224/checkpoint-14.pth.tar'
parser.add_argument('--onnx_model', default="FaceMask5.onnx")    # "mobilenetv2.onnx"    # "FaceAnti-Spoofing.onnx"
parser.add_argument('--onnx_model_sim', default="FaceMask2-sim.onnx", help='Output ONNX simple model')     # "FaceAnti-Spoofing-sim.onnx"
args = parser.parse_args()


print("=====> load pytorch checkpoint...")
mymodel = create_model(
        model_name=args.model_backbone,
        pretrained=False,
        num_classes=args.classes_num,
        in_chans=args.input_channel,
        global_pool='avg',
        checkpoint_path=args.torch_model)

# mymodel = create_model(
#         model_name='mobilenetv2_075',
#         pretrained=False,
#         num_classes=2,
#         in_chans=3,
#         global_pool='avg',
#         checkpoint_path='/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/mobilenetv2_075_no_prefetcher/mobilenetv2_075.pth.tar')  # '/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/checkpoint-36.pth.tar'  # './checkpoints/train/20200319-182337-mobilenetv2_100-224/checkpoint-14.pth.tar'


print("=====> convert pytorch model to onnx...")
# An example input you would normally provide to your model's forward() method
# x = torch.rand(1, 3, 224, 224)
x = torch.rand(1, args.input_channel, args.input_size, args.input_size)                  # for bincamera classification assignment

# Export the model
# torch_out = torch.onnx._export(mymodel, x, "mobilenetv2.onnx", export_params=True)
torch_out = torch.onnx._export(mymodel, x, args.onnx_model, export_params=True)

# cv2.waitKey(10000)

# print("=====> check onnx model...")
# model = onnx.load(args.onnx_model)
# onnx.checker.check_model(model)
# # Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)
# print(model.graph)


# print("=====> Simplifying...")
# model_opt = onnxsim.simplify(args.onnx_model)
#
#
# onnx.save(model_opt, args.onnx_model_sim)
# print("onnx model simplify OK!")