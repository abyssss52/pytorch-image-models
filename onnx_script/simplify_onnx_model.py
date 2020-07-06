import onnx
import onnxsim
import argparse

parser = argparse.ArgumentParser(description='simplify_onnx_model')
parser.add_argument('--onnx_model', default="mobilenetv2_new.onnx")    # "mobilenetv2.onnx"    # "FaceAnti-Spoofing.onnx"
parser.add_argument('--onnx_model_sim', default="mobilenetv2_new-sim.onnx", help='Output ONNX simple model')     # "FaceAnti-Spoofing-sim.onnx"
args = parser.parse_args()


print("=====> Simplifying...")
model_opt = onnxsim.simplify(args.onnx_model)


onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify OK!")