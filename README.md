# TensorRT
+ Completed: TensorRt + Tensorflow
+ TODO: TensorRT + Onnx
## Install
+ [tensorrt: 4.1. Debian Installation](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
+ Copy TensorRT libraries in ```/usr/lib/python3.5/dist-packages/``` to virtual environment libraries(site-packages), if working with a "virtual environment".
  + graphsurgeon
  + graphsurgeon-0.4.1.dist-info
  + tensorrt
  + tensorrt-5.1.5.0.dist-info
  + uff
  + uff-0.6.3.dist-info


## TensorRT with Tensorflow
+ Introduction and Code are placed in directory: "googlenet"
+ Because of library: "tensorflow.contrib.tensorrt" is deleted at tensorflow 2.0, the convert method may be different.
+ Note that TensorRT==5.1.5 may not support tensorflow 1.14, even thought it still has library: "tensorflow.contrib.tensorrt"
### Tensorflow >= 2.0
+ Seems not supported now .. 2019/7/12

### Tensorflow <= 1.13
+ If you are looking for an implement by slim, change git branch to "slim_ver" for more detail. Method of both two are similar.  

## TensorRT with Onnx


