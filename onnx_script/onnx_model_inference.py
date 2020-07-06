import math
import os
import shutil
import numpy as np

import onnx
import onnxruntime as rt

from PIL import Image
import cv2

import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.transforms import _pil_interp, ToNumpy, ToTensor


# Load the ONNX model
model = onnx.load("./FaceMask5-sim.onnx")   # "./mobilenetv2_new_sim.onnx"
# model = onnx.load("../FaceAnti-Spoofing.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model)


# onnx inference    # this is for bincamera face_anti
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


# input_data = np.ones((1, 4, 112, 112), dtype=np.float32)
file_path = '/home/night/图片/temp/face_mask/'# '/home/night/图片/temp/face_mask'# '/home/night/PycharmProjects/RKNN/mobilenetv2/test'# '/home/night/PycharmProjects/Picture_Classification/pytorch-image-models/Datasets/face_mask_test' # # '/home/night/Datasets/25_img_move'# '/home/night/Datasets/live_v2/室内活体demo_img_move'# '/home/night/Datasets/live_v2/normal_photo_fake_img_move'# '/home/night/Datasets/live_v2/面板机photo_face_anti_img_move'
face_list = os.listdir(file_path)
face_list = sorted(face_list)
# print(face_list)
correct_num = 0
cls = ['未戴口罩', '戴口罩']
transform = transforms_imagenet_eval(img_size=224)

# for i in range(int(len(face_list)/2)):
#     path = os.path.join(file_path, face_list[2*i])#'/home/night/图片/temp/spoof/data(1)/a_0000000001_0.png'
#     img_ori = Image.open(path).convert('RGB')
#     # img_ori = np.array(img_ori).astype(np.float32)
#     # print(img_ori)
#     transform = transforms_imagenet_eval(img_size=112)
#     img_ori = transform(img_ori)
#     # img_ori = np.transpose(img_ori, (2, 0, 1))
#     # img_ori = img_ori - np.array([123, 116, 103]).reshape((3, 1, 1))
#     # img_ori = img_ori / np.array([123, 116, 103]).reshape((3, 1, 1))
#     # img_ori = img_ori[None, :, :, :]
#
#     img_ori = np.array(img_ori).astype(np.float32)  # Image 格式图片转换成array
#
#     path = os.path.join(file_path, face_list[2*i+1]) #'/home/night/图片/temp/spoof/data(1)/a_0000000001_1.png'
#     img_ir = Image.open(path).convert('RGB')
#     img_ir = transform(img_ir)
#     img_ir = np.array(img_ir).astype(np.float32)  # Image 格式图片转换成array
#
#     img_ir = img_ir[0, :, :]
#     img_ir = img_ir[np.newaxis, :]
#
#     img = np.concatenate([img_ori, img_ir], axis=0)
#     input_data = img[np.newaxis, :]
#     # img = torch.from_numpy(img)
#
#     sess = rt.InferenceSession("../FaceAnti-Spoofing.onnx")
#
#     # get output name
#     input_name = sess.get_inputs()[0].name
#     # print("input name", input_name)
#     output_name= sess.get_outputs()[0].name
#     # print("output name", output_name)
#     output_shape = sess.get_outputs()[0].shape
#     # print("output shape", output_shape)
#     #forward model
#     output = sess.run([output_name], {input_name: input_data})
#     out = np.array(output)
#     print('神经网络输出：', out)
#     cls_result = np.argmax(out)
#     if cls_result == 1:
#         correct_num += 1
#
#     # if cls_result == 0:
#     #     correct_num += 1
#     # print(out)

for face in face_list:
    path = os.path.join(file_path, face)  # '/home/night/图片/temp/spoof/data(1)/a_0000000001_0.png'
    img = Image.open(path).convert('RGB')
    img_show = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # Image格式图片转换成OpenCV格式
    img_show = cv2.resize(img_show,(224,224))
    cv2.imshow('result', img_show)
    # img.show()
    img = transform(img)
    img = np.array(img).astype(np.float32)  # Image 格式图片转换成array

    input_data = img[np.newaxis, :]
    # print(input_data.shape)
    sess = rt.InferenceSession("./FaceMask3_sim.onnx")

    # get output name
    input_name = sess.get_inputs()[0].name
    # print("input name", input_name)
    output_name = sess.get_outputs()[0].name
    # print("output name", output_name)
    output_shape = sess.get_outputs()[0].shape
    # print("output shape", output_shape)
    # forward model
    output = sess.run([output_name], {input_name: input_data})
    out = np.array(output)
    print('神经网络输出：', out)
    print(cls[np.argmax(out)])

    if cv2.waitKey(0) == ord('n'):
        pass
    # cls_result = np.argmax(out)
    # if cls_result == 0:
    #     correct_num += 1
    # else:
    #     with open('./incorrect_image.txt', 'a') as f:
    #         f.write(face)
    #         f.write('\n')
        # shutil.move(os.path.join(file_path, face), '/home/night/Datasets/face/face_mask/incorrect_image')

# print('判断为没戴口罩人脸的次数：', correct_num)
# print('判断为戴口罩人脸的次数：', correct_num)
# print('判断为假人脸的次数：', correct_num)
# print('判断为真人脸的次数：', correct_num)