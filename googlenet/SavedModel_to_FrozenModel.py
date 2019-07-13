# from SavedModel to frozen graph
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import freeze_graph

from googlenet.checkpoint_to_SavedModel import image_web_saved_encode


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


SavedModel_dir = "./SavedModel/cnn_model/"
SavedModel_path = os.path.join(SavedModel_dir, str(len(os.listdir(SavedModel_dir))-2))

summaries_dir = "./frozen_model/cnn_model/tensorboard/"
directory_create(summaries_dir)
frozen_export_model_dir = "./frozen_model/cnn_model/"
frozen_export_model_dir = os.path.join(frozen_export_model_dir, str(len(os.listdir(frozen_export_model_dir))-1))

batch_size = 1
max_GPU_mem_size_for_TRT = 2 << 20

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
        tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], SavedModel_path)

        tf.train.write_graph(sess.graph_def, frozen_export_model_dir, 'model.pb')
        freeze_graph.freeze_graph(
            input_graph=None,
            input_saver=None,
            input_binary=False,
            input_saved_model_dir=SavedModel_path,
            input_checkpoint=None,
            output_node_names="logits",
            restore_op_name=None,  # 'save/restore_all',
            filename_tensor_name=None,  # 'save/Const:0',
            output_graph=os.path.join(frozen_export_model_dir, 'frozen_model.pb'),
            clear_devices=False,
            initializer_nodes=''
        )
        print("frozen done")

# =========================================================

print("Start Evaluate")
# load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
with tf.gfile.GFile(os.path.join(frozen_export_model_dir, 'frozen_model.pb'), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# import the graph_def into a new Graph and returns it
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
        # warm up
        print("warm up")
        for i in range(5):
            prob = sess.run("prefix/logits:0", {
                "prefix/image_strings:0": [x_test[0]] * batch_size,
                "prefix/image_shapes:0": [(28, 28, 3)] * batch_size
            })
        print("counter start")
        START_TIME = time.time()
        prob = sess.run("prefix/logits:0", feed_dict={
            "prefix/image_strings:0": [x_test[0]] * batch_size,
            "prefix/image_shapes:0": [(28, 28, 3)] * batch_size
        })
        print("spent %f seconds" % (time.time() - START_TIME))
        print("label: %d, prediction: %d" % (np.argmax(y_test[0]), np.argmax(prob[0])))

        # write graph
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir, graph)
