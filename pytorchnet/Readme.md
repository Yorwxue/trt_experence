# Apply pytorch to Nvidia-TensorRT-Inference-Server

## Convert Pytorch to TensorRT via Onnx
+ We using pytorch 0.4.1 as example
+ If you using pytorch >= 1.10, you can following there [steps](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

### Conver Pytorch Model to Onnx
+ [How to convert pytorch model to onnx](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
+ [pytorch to mxnet/CNTK via onnx](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-onnx-pytorch-mxnet.html)
+ Note that torch_onnx CANNOT export model in cuda to onnx, so it's necessary to do ```model.to("cpu")```

```python
from torch.autograd import Variable
import torch.onnx as torch_onnx

YOUR_MODEL.to("cpu")
dummy_input = Variable(torch.randn(1, *INPUT_SHAPE))
output = torch_onnx.export(YOUR_MODEL,
                          dummy_input,
                          onnx_model_path,
                          verbose=False)
```

+ [Optimize Pytorch Model](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#converting-opset-version-of-an-onnx-model)

### Running with ONNXRUNTIME

### Conver Onnx to TensorRT

## Convert Pytorch to TensorRT Directory
+ [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
+ May only work in pytorch >= 1.1.0
