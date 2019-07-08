# Import TensorFlow and TensorRT
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

SavedModel_dir = "./pretrained/SavedModel/inception_resnet_v2/"
model_tag = "serve"  # can be found by saved_model_cli
SavedModel_path = os.path.join(
    SavedModel_dir,
    max(os.listdir(SavedModel_dir))
)
print("model path: ", SavedModel_path)
trt_model_dir = "./pretrained/trt/inception_resnet_v2/"

batch_size = 2
max_GPU_mem_size_for_TRT = 2 << 20

# Inference with TF-TRT `SavedModel` workflow:
graph = tf.Graph()
with graph.as_default():
    # tfconfig = tf.ConfigProto()
    # tfconfig.gpu_options.allow_growth = True  # maybe necessary
    # tfconfig.allow_soft_placement = True  # maybe necessary
    # with tf.Session(config=tfconfig) as sess:
    with tf.Session() as sess:
        # Create a TensorRT inference graph from a SavedModel:
        trt_graph = trt.create_inference_graph(
            input_graph_def=None,
            outputs=None,
            input_saved_model_dir=SavedModel_path,
            input_saved_model_tags=[model_tag],
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_GPU_mem_size_for_TRT,
            precision_mode="FP32",
            output_saved_model_dir=trt_model_dir
        )
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=["probs:0"]
        )
        sess.run(output_node)
