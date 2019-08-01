import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torch.onnx as torch_onnx
import onnx

from pytorchnet.dataloader import ExampleData

torch.cuda.manual_seed(19)


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv_net = torch.nn.Sequential()
        self.conv_net.add_module("conv2d_1", torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3)))
        self.conv_net.add_module("conv2d_1_max_pool2d", torch.nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv_net.add_module("conv2d_1_relu", torch.nn.ReLU())
        self.conv_net.add_module("conv2d_2", torch.nn.Conv2d(5, 2, (3, 3)))
        self.conv_net.add_module("conv2d_2_dropout", torch.nn.Dropout2d())
        self.conv_net.add_module("conv2d_2_max_pool2d", torch.nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv_net.add_module("conv2d_2_relu", torch.nn.ReLU())
        # self.add_module("conv2d_2_flatten", torch.nn.)
        self.fc_net = torch.nn.Sequential()
        self.fc_net.add_module("fc1", torch.nn.Linear(1058, 100))
        self.fc_net.add_module("fc1_dropout", torch.nn.Dropout2d())
        self.fc_net.add_module("fc2", torch.nn.Linear(100, 3))
        self.fc_net.add_module("softmax", torch.nn.Softmax())

    def forward(self, x):

        x = x.type(torch.float32)
        x = torch.div(x, 255.)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_net(x)
        # x = x.view(x.size()[0], -1)  # this will cause an error in conversion from pytorch to onnx
        x = x.view([int(x.size()[0]), -1])  # temporary solution
        # x = x.flatten(1)  #
        x = self.fc_net(x)
        return x


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
input_shape = (100, 100, 3)
batch_size = 2
num_workers = 0
num_of_data = batch_size * 5

# using cuda
device = torch.device("cuda")

# example data
example_dataloader = ExampleData(num=num_of_data, data_shape=input_shape)
dataloader = DataLoader(example_dataloader, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# create model
model = Model()

# load model
# if os.path.exists(torch_model_path):
#     model.model = torch.load(torch_model_path)

# device
model.to(device)

# eval
model.train(False)
model.eval()

for batch_idx, data in enumerate(tqdm(dataloader), 1):
    x = data["input"]
    y = data["label"]
    x, y = x.to(device, dtype=torch.int32), y.to(device)

    # forward
    output = model(x)
    # print("output: ", output)

    # get the index of the max log-probability
    for idx in range(list(output.shape)[0]):
        # pred = output[idx].max(0)
        pred = output[idx]
        print(x[idx, 0, :2, :2])
        print("pred_%d: " % idx, pred[idx])
        print("==========================================================")

# save model
torch.save(model, torch_model_path)

# Export the model to an ONNX file
if os.path.exists(onnx_model_path):
    os.system("rm %s" % onnx_model_path)
model.to("cpu")  # It's necessary!! torch_onnx CANNOT export model in cuda to onnx
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(model,
                           dummy_input,
                           onnx_model_path,
                           input_names=["input_1"],
                           output_names=["output_1"],
                           verbose=True)
print("Export of torch_model.onnx complete!")

print("====================================")
print("Architecture in onnx:")
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
arch = onnx.helper.printable_graph(onnx_model.graph)
print(arch)
