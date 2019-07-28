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

## Apply Onnx in Nvidia-TensorRT-Inference-Server
+ [official doc.](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/quickstart.html)
+ Introduction can be found in "trt_experiment/trt_inference_server"

## ONNX Runtime Server
+ TODO
+ [official doc.](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles)
