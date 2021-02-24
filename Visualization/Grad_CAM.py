import torch
import torchvision
from torchvision import models
from torch.autograd import Function
from torchsummary import summary

from timm.models import create_model

import argparse
import cv2
import numpy as np
import os



# parser = argparse.ArgumentParser(description='pytorch CNN visualization by Grad-CAM')
# parser.add_argument('--model_backbone',type=str, default='mobilenetv2_100',help='the type of model chosen.')
# parser.add_argument('--classes_num', type=int, default=2, help='the number of classification class.')
# parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel.')
# parser.add_argument('--input_size', type=int, default=224, help='the size of input.')
# parser.add_argument('--torch_model', default='/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/mobilenetv2_100_no_prefetcher/checkpoint-42.pth.tar')   # "/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/Live_Detection/model_best.pth.tar" #   # './checkpoints/train/20200319-182337-mobilenetv2_100-224/checkpoint-14.pth.tar'
#
# args = parser.parse_args()
#
# print("=====> load pytorch checkpoint...")
# mymodel = create_model(
#         model_name=args.model_backbone,
#         pretrained=False,
#         num_classes=args.classes_num,
#         in_chans=args.input_channel,
#         global_pool='avg',
#         checkpoint_path=args.torch_model)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# mymodel = mymodel.to(device)
#
# # mymodel = models.resnet50(pretrained=False)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# # mymodel = mymodel.to(device)
#
# # summary(mymodel, input_size=(args.input_channel, args.input_size, args.input_size))
# for name, module in mymodel._modules.items():
#     print(name)
#     print(module)
#     print('=============================>')
# # print(mymodel._modules.items())


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "global_pool" in name.lower():   # avgpool
                x = module(x)
                x = x.view(x.size(0), -1)
            elif "avgpool" in name.lower():   # avgpool
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imshow('result', np.uint8(255 * cam))
    if cv2.waitKey(0) == ord('n'):
        pass
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        result = ['face', 'mask']
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            print(output)
            print('这是:', result[index])

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, result[index]


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser(description='pytorch CNN visualization by Grad-CAM')
    parser.add_argument('--model_backbone', type=str, default='efficientnet_lite0', help='the type of model chosen.')   #  mnasnet_small semnasnet_100  # mobilenetv2_100  # shufflenetv2_100
    parser.add_argument('--classes_num', type=int, default=2, help='the number of classification class.')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel.')
    parser.add_argument('--input_size', type=int, default=224, help='the size of input.')
    parser.add_argument('--torch_model',
                        default='/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/EfficientNet-lite0/checkpoint-88.pth.tar')    #  '/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/mobilenetv2_100_no_prefetcher/checkpoint-34.pth.tar' # '/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/ShuffleNetv2_100/checkpoint-59.pth.tar' # '/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/face_mask/MobileNeXt_100/checkpoint-84.pth.tar'  # "/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/checkpoints/Live_Detection/model_best.pth.tar"
    parser.add_argument('--use-cuda', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/home/night/abyss52/work/Test_data/face_mask/beard',
                        help='Input image path')    # '/home/night/Datasets/face/face_mask/val/face'    #  '/home/night/abyss52/work/Test_data/face_mask/test/1'  # '/home/night/abyss52/work/Test_data/face_mask/tmp'
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    print("=====> load pytorch checkpoint...")
    mymodel = create_model(
        model_name=args.model_backbone,
        pretrained=False,
        num_classes=args.classes_num,
        in_chans=args.input_channel,
        global_pool='avg',
        checkpoint_path=args.torch_model)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    mymodel = mymodel.to(device)

    # mymodel = models.resnet50(pretrained=False)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    # mymodel = mymodel.to(device)

    summary(mymodel, input_size=(args.input_channel, args.input_size, args.input_size))

    grad_cam = GradCam(model=mymodel, feature_module=mymodel.final_layers, target_layer_names=["0"], use_cuda=args.use_cuda)

    wrong_count = 0

    # 循环显示图片
    for img_name in os.listdir(args.image_path):
        img_path = os.path.join(args.image_path, img_name)
        print(img_path)
        img_ori = cv2.imread(img_path)
        img = np.float32(cv2.resize(img_ori, (224, 224))) / 255
        input = preprocess_image(img)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask, cls = grad_cam(input, target_index)

        # show_cam_on_image(img, mask)

        if cls == 'mask':
            wrong_count += 1

        # cv2.imwrite(os.path.join('/home/night/abyss52/work/Dataset/face/Celeba/img_Celeba/', cls, img_name), img_ori)

        # gb_model = GuidedBackpropReLUModel(model=mymodel, use_cuda=args.use_cuda)
        # # print(mymodel._modules.items())
        # gb = gb_model(input, index=target_index)
        # gb = gb.transpose((1, 2, 0))
        # cam_mask = cv2.merge([mask, mask, mask])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)
        #
        # cv2.imwrite('gb.jpg', gb)
        # cv2.imwrite('cam_gb.jpg', cam_gb)


    print('错误结果数：', wrong_count)