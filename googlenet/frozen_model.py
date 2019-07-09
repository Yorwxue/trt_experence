import importlib
import time
import os

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.tools import freeze_graph
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import skimage
import skimage.io
import skimage.transform

# -------------------------
try:
    from data.imagenet_classes import *
except Exception as e:
    raise Exception("{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


input_size_h = 299
input_size_w = 299
# pretrained_model_path = "./pretrained/inception_resnet_v2_2016_08_30.ckpt"
# pretrained_model_path = "./pretrained/inception_v3.ckpt"

checkpoint_path = "./pretrained/checkpoint/inception_resnet_v2/"
# checkpoint_path = "./pretrained/checkpoint/inception_v3/"

frozen_model_dir = "./pretrained/frozen_model/inception_resnet_v2/"
directory_create(frozen_model_dir)


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 299, 299
    resized_img = skimage.transform.resize(crop_img, (input_size_h, input_size_w))
    return resized_img


def print_prob(prob):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


if __name__ == "__main__":
    # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    img1 = load_image("data/dog/img1.jpg")
    img1 = img1.reshape((1, input_size_h, input_size_w, 3))

    image_batch = tf.placeholder(tf.float32, shape=(None, input_size_h, input_size_w, 3), name="image_batch")

    # inception_v3
    """
    network = importlib.import_module('models.inception_v3')
    with slim.arg_scope(inception_v3_arg_scope()):
        prelogits, end_points = inception_v3(
                    image_batch, is_training=False, dropout_keep_prob=1.0,
                    num_classes=1001, reuse=None)

    probs = tf.nn.softmax(prelogits, name="probs")
    # """

    # inception_resnet_v2
    # """
    network = importlib.import_module('models.inception_resnet_v2')
    scope = network.inception_resnet_v2_arg_scope(
                weight_decay=0.0,
                batch_norm_decay=0.995,
                batch_norm_epsilon=0.001,
                activation_fn=tf.nn.relu
            )
    with slim.arg_scope(scope):
        prelogits, _ = network.inception_resnet_v2(
            image_batch, is_training=False, dropout_keep_prob=1.0,
            num_classes=1001, reuse=None)

    probs = tf.nn.softmax(prelogits, name="probs")
    # """

    sess = tf.InteractiveSession()

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # model restore
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    print("Model Restored")

    # print_tensors_in_checkpoint_file(checkpoint_path, all_tensors=False, tensor_name='')

    # forward pass
    start_time = time.time()
    prob = sess.run(probs, feed_dict={image_batch: img1})
    print("End time : %.5ss" % (time.time() - start_time))
    print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing

    tf.train.write_graph(sess.graph_def, frozen_model_dir, 'model.pb')

    freeze_graph.freeze_graph(
        input_graph=os.path.join(frozen_model_dir, 'model.pb'),
        input_saver='',
        input_binary=False,
        input_checkpoint=checkpoint_path,
        output_node_names="probs",
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=os.path.join(frozen_model_dir, 'frozen_model.pb'),
        clear_devices=False,
        initializer_nodes=''
    )
    print("done")
