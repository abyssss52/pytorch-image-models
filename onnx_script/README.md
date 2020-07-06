# The Tool for Pytorch2Onnx and Onnx Inference

## Requirement
Before use this code, install package `timm`.This package can be installed via pip. Currently, the model factory (`timm.create_model`) is the most useful component to use via a pip install.

Install (after conda env/install):
```
pip install timm
```

Use:
```
>>> from timm import create_model
>>> m = create_model('mobilenetv3_100', pretrained=True)
```

other library file:`torch, torchvision, argparse, onnx, onnxsim`
