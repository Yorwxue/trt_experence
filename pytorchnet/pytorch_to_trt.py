# torch2trt can only used in global environment now
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch2trt import torch2trt

from pytorchnet.dataloader import ExampleData


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(5, 2, (3, 3))
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(1058, 100)
        self.fc2 = torch.nn.Linear(100, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(x, (2, 2))
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = torch.nn.functional.max_pool2d(x, (2, 2))
        x = torch.nn.functional.relu(x)
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = TestNet()

    def forward(self, inputs):
        y = self.model(inputs)
        return y


# model path
torch_model_dir = os.path.abspath(os.path.join(__file__, "..", "torch_model"))
torch_model_name = "torch_model"
torch_model_path = os.path.join(torch_model_dir, torch_model_name)
onnx_model_dir = os.path.abspath(os.path.join(__file__, "..", "torch_onnx_model"))
onnx_model_name = "torch_onnx_model.onnx"
onnx_model_path = os.path.join(onnx_model_dir, onnx_model_name)
if not os.path.exists(torch_model_dir):
    os.makedirs(torch_model_dir)
if not os.path.exists(onnx_model_dir):
    os.makedirs(onnx_model_dir)

# data parameter
input_shape = (3, 100, 100)
batch_size = 2
num_workers = 1
num_of_data = batch_size * 5

# using cuda
device = torch.device("cuda")

# example data
example_dataloader = ExampleData(num=num_of_data, data_shape=input_shape)
dataloader = DataLoader(example_dataloader, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# create model
model = Model()

# load model
if os.path.exists(torch_model_path):
    model.model = torch.load(torch_model_path)

# device
model.to(device)

# eval
model.train(False)
model.eval()

# for batch_idx, data in enumerate(tqdm(dataloader), 1):
#     x = data["input"]
#     y = data["label"]
#     x, y = x.to(device, dtype=torch.float32), y.to(device)
#     print(x[0, 0:2, 0:2, 0])
#
#     # forward
#     output = model(x)
#     # print("output: ", output)
#
#     # get the index of the max log-probability
#     for idx in range(list(output.shape)[0]):
#         pred = output[idx].max(0)
#         print("pred_%d: " % idx, pred)

# save model
# torch.save(model, torch_model_path)

# create example data
ex_data = torch.ones((1, *input_shape)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [ex_data])

# test
np.random.seed(19)
data = np.random.randn(num_of_data, *input_shape)
y_trt = model_trt(data[0])
print(y_trt)
