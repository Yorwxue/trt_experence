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
#### Load Model, and Parsing:
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
+ Mark output for TensorRT in parsing processing before building engine(not necessary for trt >= 5.1.5).

#### Inference
[official](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_python)
+ Asking for memory in gpu
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0))*batch_size, dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1))*batch_size, dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
```
+ engine is created in the session ["Load model, and parsing"](#Load-Model,-and-Parsing)
+ If encounter an error like: "pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?", you should "import pycuda.autoinit"

+ Data Copy from host to device
+ you should notice the size of data, it will cause error in result, but with no warming.

```python
# unsynchronized
stream = cuda.Stream()
cuda.memcpy_htod_async(d_input, YOUR_INPUT_DATA, stream)

# synchronized
cuda.memcpy_htod(d_input, YOUR_INPUT_DATA)

```

+ Model Computing
```python
# unsynchronized
context.execute_async(YOUR_BATCH_SIZE, [int(d_input), int(d_output)], stream.handle, None)

# synchronized
context.execute(YOUR_BATCH_SIZE, bindings=[int(d_input), int(d_output)])
```

+ Get result from device to host
```python
# unsynchronized
stream.synchronize()
cuda.memcpy_dtoh_async(h_output, d_output, stream)

# synchronized
cuda.memcpy_dtoh(h_output, d_output)
```

### Onnx-TensorRT
+ [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)
+ TODO

## Apply Nvidia-TensorRT-Inference-Server
+ [official doc.](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/quickstart.html)
+ For some model it's necessary to create an model script
### Create Model Script
+ [Model Script](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_repository.html#tensorrt-models) is necessary for inference-server.
```
name: "Model_NAME"
platform: "tensorrt_plan"
max_batch_size: 10
input [
  {
    name: "INPUT_NAME"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [3, 100, 100]
  }
]
output [
  {
    name: "OUTPUT_NAME"
    data_type: TYPE_FP32
    dims: [3]
  }
]
instance_group [
  {
    kind: KIND_GPU,
    count: 1
  }
]
```
+ If your tensorrt model convert along "pytorch -> onnx -> tensorrt", Parameter: INPUT_NAME and OUTPUT_NAME need to be marked when convert pytorch to onnx.

### Inference Server
+ Note that you don't need to encode image as base64 string, you just need to convert to bytes, and send request to server.

#### Start Server
```bash
$ NV_GPU=0  docker run --runtime=nvidia --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /mnt/hdd1/trt_experiment/onnx_trt_model/pytorchnet/:/models nvcr.io/nvidia/tensorrtserver:19.06-py3 trtserver --model-store=/models
```

#### Check Server
```bash
$ curl localhost:8000/api/status
```

#### Inference
##### By Python Code:
```python
import requests
import cv2

image = cv2.imread("/PATH/TO/YOUR/IMAGE")
array_img = image.astype("float32")
bytes_img = array_img.tobytes()
batch_size = 1

headers = {
    "NV-InferRequest": 'batch_size: %d input { name: "input_1" dims: 100 dims: 100 dims: 3} output { name: "output_1"  cls { count: 3 } }' % batch_size,
}
url = "http://localhost:8000/api/infer/Model_NAME"
response = requests.request("POST", url, data=bytes_img, headers=headers)
```
+ If you want to inference batch-by-batch, you can concatenate two bytes_image, and set batch_size to YOUR_BATCH_SIZE.
```python
bytes_imgs = bytes_img + bytes_img
```

##### By Curl:
+ [TODO](https://github.com/NVIDIA/tensorrt-inference-server/issues/280)


## ONNX Runtime Server
+ TODO
+ [official doc.](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles)
