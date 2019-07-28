import os
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import onnx
from onnx import version_converter
from onnx import optimizer, shape_inference
import onnxruntime as rt

from pytorchnet.dataloader import ExampleData


onnx_model_dir = os.path.abspath(os.path.join(__file__, "..", "torch_onnx_model/"))
onnx_model_path = os.path.join(onnx_model_dir, "torch_onnx_model.onnx")

# data parameter
input_shape = (3, 100, 100)
batch_size = 2
num_workers = 1
num_of_data = batch_size * 5

# preparing dataset
# """
example_dataloader = ExampleData(num=num_of_data, data_shape=input_shape)
dataloader = DataLoader(example_dataloader, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# """

# onnx
print("onnx checking .. ", end='')
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ok")

# print("version convert .. ", end='')
# not work
# converted_model = version_converter.convert_version(onnx_model, 7)
# print("ok")

# optimizing an onnx model
# """
# # A full list of supported optimization passes can be found using get_available_passes()
# all_passes = optimizer.get_available_passes()
# print("Available optimization passes:")
# for p in all_passes:
#     print(p)
# print()

# Pick one pass as example
passes = ['fuse_consecutive_transposes']

# Apply the optimization on the original model
optimized_model = optimizer.optimize(onnx_model, passes)
# print('The model after optimization:\n{}'.format(optimized_model))

# One can also apply the default passes on the (serialized) model
# Check the default passes here: https://github.com/onnx/onnx/blob/master/onnx/optimizer.py#L43
# optimized_model = optimizer.optimize(onnx_model)
# """

# Apply shape inference on the model
inferred_model = shape_inference.infer_shapes(optimized_model)

# Check the model and print Y's shape information
onnx.checker.check_model(inferred_model)
# print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))

print("create session .. ", end='')
model_str = inferred_model.SerializeToString()
# sess = rt.InferenceSession(onnx_model_path)
sess = rt.InferenceSession(model_str)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print("ok")

START_TIME = time.time()
for batch_idx, data in enumerate(tqdm(dataloader), 1):
    x = data["input"]
    y = data["label"]
    pred_onx = sess.run([label_name], {
        input_name: np.asarray(x)
    })[0]
    print("output: ", pred_onx)
print("spent %f seconds" % (time.time() - START_TIME))

