# from frozen model to Onnx
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf2onnx
import os


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


input_height = 28
input_width = 28

summaries_dir = "./onnx_model/cnn_model/tensorboard/"
directory_create(summaries_dir)
onnx_export_dir = "./onnx_model/cnn_model/"
onnx_export_dir = os.path.join(onnx_export_dir, str(len(os.listdir(onnx_export_dir))-1))
directory_create(onnx_export_dir)
onnx_export_path = os.path.join(onnx_export_dir, "model.onnx")

# preparing dataset
# """
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# reshape from 784 to 28*28
# x_train = np.reshape(x_train, [x_train.shape[0], 28, 28, 1])
x_test = np.reshape(x_test, [x_test.shape[0], input_height, input_width, 1])

# base64 encode
# x_train = [image_web_saved_encode(np.concatenate([image, image, image], axis=2)*255) for image in list(x_train)]
# x_test = [image_web_saved_encode(np.concatenate([image, image, image], axis=2)*255) for image in list(x_test)]

# fill 3 channels with copy
x_test = [np.concatenate([image, image, image], axis=2) for image in list(x_test)]
# """


# SavedModel to onnx
"""
SavedModel_dir = os.path.abspath("./SavedModel/cnn_model/")
SavedModel_path = os.path.join(SavedModel_dir, str(len(os.listdir(SavedModel_dir))-2))

# parameter of GPU
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True  # maybe necessary
tfconfig.allow_soft_placement = True  # maybe necessary
# tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(graph=tf.Graph(), config=tfconfig) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.SERVING],
                               SavedModel_path)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                                 input_names=[
                                                     "image_strings:0",
                                                     "image_shapes:0"
                                                 ],
                                                 output_names=["logits:0"])
    model_proto = onnx_graph.make_model("test")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
# """


# import the graph_def into a new Graph and returns it
convert = True
if convert:
    # frozen model to onnx
    frozen_dir = os.path.abspath("./frozen_model/cnn_model/")
    frozen_dir = os.path.join(frozen_dir, str(len(os.listdir(frozen_dir)) - 2))
    print("frozen model path: %s" % frozen_dir)

    # load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(os.path.join(frozen_dir, 'frozen_model.pb'), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import the graph_def into a new Graph
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")

        for op in graph.get_operations():
            print(op.name)

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True  # maybe necessary
        tfconfig.allow_soft_placement = True  # maybe necessary
        with tf.Session(config=tfconfig) as sess:
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                                         input_names=[
                                                             "prefix/image_batch:0",
                                                             # "prefix/image_strings:0",
                                                             # "prefix/image_shapes:0"
                                                         ],
                                                         output_names=["prefix/logits:0"])
            model_proto = onnx_graph.make_model("test")
            with open(onnx_export_path, "wb") as f:
                f.write(model_proto.SerializeToString())
                print("model saved at %s" % onnx_export_path)

