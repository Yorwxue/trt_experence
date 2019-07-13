import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.examples.tutorials.mnist import input_data

from googlenet.SavedModel_loader import image_web_saved_encode


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


summaries_dir = "./trt_model/cnn_model/tensorboard/"
directory_create(summaries_dir)
SavedModel_dir = "./SavedModel/cnn_model/"
SavedModel_path = os.path.join(SavedModel_dir, str(len(os.listdir(SavedModel_dir))))
model_tag = "serve"  # can be queried by saved_model_cli
batch_size = 1
max_GPU_mem_size_for_TRT = 2 << 20
trt_model_dir = "./trt_model/cnn_model/"
trt_model_dir = os.path.join(trt_model_dir, str(len(os.listdir(trt_model_dir))))

# preparing dataset
# """
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# reshape from 784 to 28*28
# x_train = np.reshape(x_train, [x_train.shape[0], 28, 28, 1])
x_test = np.reshape(x_test, [x_test.shape[0], 28, 28, 1])

# base64 encode
# x_train = [image_web_saved_encode(np.concatenate([image, image, image], axis=2)*255) for image in list(x_train)]
x_test = [image_web_saved_encode(np.concatenate([image, image, image], axis=2) * 255) for image in list(x_test)]
# """

# Inference with TF-TRT `SavedModel` workflow:
# """
graph = tf.Graph()
with graph.as_default():
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # maybe necessary
    tfconfig.allow_soft_placement = True  # maybe necessary
    with tf.Session(config=tfconfig) as sess:
        # Create a TensorRT inference graph from a SavedModel:
        trt_graph = trt.create_inference_graph(
            input_graph_def=None,
            outputs=None,
            # is_dynamic_op=True,
            input_saved_model_dir=SavedModel_path,
            input_saved_model_tags=[model_tag],
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_GPU_mem_size_for_TRT,
            precision_mode="FP32",
            # use_calibration=False,  # set False when using INT8
            # The following command will create a directory automatically,
            # and you must notice that "output_saved_model_dir" need to specific a path without point to any directory
            output_saved_model_dir=None  # trt_model_dir
        )
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=["logits:0"]
        )

        trt_engine_ops = [n.name for n in trt_graph.node if str(n.op) == 'TRTEngineOp']
        print("Number of trt op: %d" % len(trt_engine_ops))
        print(trt_engine_ops)

        # warm up
        print("warm up")
        for i in range(5):
            prob = sess.run(output_node, {
                "import/image_strings:0": [x_test[0]] * batch_size,
                "import/image_shapes:0": [(28, 28, 3)] * batch_size
            })
        print("counter start")
        START_TIME = time.time()
        prob = sess.run(output_node, feed_dict={
            "import/image_strings:0": [x_test[0]] * batch_size,
            "import/image_shapes:0": [(28, 28, 3)] * batch_size
        })
        print("spent %f seconds" % (time.time() - START_TIME))

        test_idx = 0
        print("label: %d, prediction: %d" % (np.argmax(y_test[test_idx]), np.argmax(prob[0])))

        # write graph
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir, trt_graph)
# """

# Inference with TF-TRT frozen graph workflow:
"""
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # First deserialize your frozen graph:
        frozen_model_path = os.path.join(frozen_model_dir, 'frozen_model.pb')
        with tf.gfile.GFile(frozen_model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        trt_graph = trt.create_inference_graph(
            input_graph_def=graph_def,
            outputs=["probs:0"],
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_GPU_mem_size_for_TRT,
            precision_mode="FP32")
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=["probs:0"])
        sess.run(output_node, feed_dict={
            "image_batch:0": img1
        })
# """

