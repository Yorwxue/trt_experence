## Using TensorRT in TensorFlow (TF-TRT)
+ Or you can use tensorflow-onnx as mention in follow [session](# Using tensorflow-onnx)
+ Note that the following example can only be used in "tensorflow<=1.13", due to tf.contrib library is deleted in tensorflow 2.0.
### Inference with TF-TRT `SavedModel` workflow:
+ using instructure: create_inference_graph to create trt graph
``` python
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

SavedModel_path = "/your/path/to/SavedModel"
trt_model_dir = "/your/path/to/save/trt/graph"
model_tag = "serve"  # can be queried by saved_model_cli
batch_size = 10
max_GPU_mem_size_for_TRT = 2 << 20

graph = tf.Graph()
with graph.as_default():
    # tfconfig is used to avoid cudnn initialize error
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tfconfig.allow_soft_placement = True
    with tf.Session(config=tfconfig) as sess:
        # Create a TensorRT inference graph from a SavedModel:
        trt_graph = trt.create_inference_graph(
            input_graph_def=None,
            outputs=None,
            input_saved_model_dir=SavedModel_path,
            input_saved_model_tags=[model_tag],
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_GPU_mem_size_for_TRT,
            precision_mode="FP32",
            # The following command will create a directory automatically,
            # and you must notice that "output_saved_model_dir" need to specific a path without point to any directory
            output_saved_model_dir=trt_model_dir
        )
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=["logits:0"]
        )
        prob = sess.run(output_node, feed_dict={
            "import/image_strings:0": X_LIST,
            "import/image_shapes:0": X_SHAPE_LIST
        })
```
+ The command command: output_saved_model_dir will create a directory automatically, and you must notice that "output_saved_model_dir" need to ==specific a path without point to any directory==.
+ Names of output nodes will be included in namescope: "import", like the format aforementioned: =="import/image_strings:0"== (everynode name will be appended ":0" automatically).
+ Introduction of precision_mode can be found [here](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)

## Speedup
+ Note that trt graph need to warmup.
### Show How many Nodes Are Convert to trt-op
``` python
trt_engine_ops = [n.name for n in trt_graph.node if str(n.op)=='TRTEngineOp']
print("Number of trt op: %d" % len(trt_engine_ops))
for i in trt_engine_ops: print(i)
```
### Using TensorBoard to Visualize Optimized Graphs
[TensorRT information](https://medium.com/tensorflow/speed-up-tensorflow-inference-on-gpus-with-tensorrt-13b49f3db3fa)
[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
``` python
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir,trt_graph)
```

## Using tensorflow-onnx
+ Note that tf2onnx can only convert frozen graph to onnx
+ tensorflow op DecodeBase64 is not supported, more detail of supported op can be found [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
+ An example of code of convert tf to onnx is list as following, you can check the official document [here](https://github.com/onnx/tensorflow-onnx#creating-custom-op-mappings-from-python)
```python 
onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                             input_names=[
                                                 "prefix/image_batch:0",
                                             ],
                                             output_names=["prefix/logits:0"])
model_proto = onnx_graph.make_model("test")
with open(onnx_export_path, "wb") as f:
    f.write(model_proto.SerializeToString())
    print("model saved at %s" % onnx_export_path)
```

## Convert onnx back to tensorflow
+ TODO
+ More detail can be found [here](https://github.com/onnx/onnx-tensorflow)