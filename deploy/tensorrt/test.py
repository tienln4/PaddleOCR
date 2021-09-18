import cv2
import os
import onnxruntime
# img_dir = "valid/images"
# for fn in os.listdir(img_dir):
#     fp =os.path.join(img_dir, fn)
#     img = cv2.imread(fp)
#     H, W, C = img.shape
#     r = W/H
#     if r < 3.2 and r > 2.9:
#         img = cv2.resize(img, (100, 32))
#         cv2.imwrite(os.path.join("X/images", fn), img)

# list_img = os.listdir("X/images")
# gt = open("X/gt.txt", "a")
# lb = open("valid/gt.txt")
# for line in lb:
#     line_ = line.split("\t")
#     img_name = line_[0].split('/')[1]
#     if img_name in list_img:
#         gt.write(line)

import onnxruntime as ort
import onnx
# from caffe2.python.onnx import backend
import numpy as np

onnx_path = "/data/tienln/workspace/vehicle-analysis/models/lpr/inference.onnx"

predictor = onnx.load(onnx_path)
# onnx.checker.check_model(predictor)
# onnx.helper.printable_graph(predictor.graph)
# predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
# input_name = ort_session.get_inputs()[0].name
# result_path = "./detect_imgs_results_onnx"


# orig_image = cv2.imread("")
# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (320, 240))
# image_mean = np.array([127, 127, 127])
# image = (image - image_mean) / 128
# image = np.transpose(image, [2, 0, 1])
# image = np.expand_dims(image, axis=0)
# image = image.astype(np.float32)
# x = ort_session.run(None, {input_name: image})