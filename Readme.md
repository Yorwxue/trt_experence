# TensorRT
## Install
+ [tensorrt: 4.1. Debian Installation](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
+ Copy TensorRT libraries in ```/usr/lib/python3.5/dist-packages/``` to virtual environment libraries(site-packages), if working with a ==virtual environment==.
  + graphsurgeon
  + graphsurgeon-0.4.1.dist-info
  + tensorrt
  + tensorrt-5.1.5.0.dist-info
  + uff
  + uff-0.6.3.dist-info

## Using TensorRT in TensorFlow (TF-TRT)
+ Introduction can be found in "trt_experiment/googlenet"

## Using TensorRT in Pytorch
+ Note that here we using pytorch 0.4.1 as example

## Convert Onnx to TensorRT
### TensorRT
+ [official](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
+ Mark output for TensorRT in parsing processing before building engine as following:
```python
import tensorrt as trt
TRT_LOGGER = trt.Logger()
onnx_file_path = "/PATH/TO/YOUR/ONNX/FILE"
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = 1 << 30  # 1GB
    builder.max_batch_size = 1
    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())
    network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
    engine = builder.build_cuda_engine(network)
```
+ Inference
[official](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_python)

### Onnx-TensorRT
+ [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)

## Apply Nvidia-TensorRT-Inference-Server
+ [official doc.](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/quickstart.html)
+ Introduction can be found in "trt_experiment/trt_inference_server"

## ONNX Runtime Server
+ TODO
+ [official doc.](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles)
