import importlib
import time
import os

import numpy as np
import tensorflow as tf
import cv2
import pybase64 as base64

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope
import skimage
import skimage.io
import skimage.transform

# -------------------------
try:
    from data.imagenet_classes import *
except Exception as e:
    raise Exception(
        "{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# image size for google-net
input_size_h = 299
input_size_w = 299
pretrained_model_path = "./pretrained/inception_resnet_v2_2016_08_30.ckpt"
# pretrained_model_path = "./pretrained/inception_v3.ckpt"

# checkpoint_path = "./pretrained/checkpoint/inception_resnet_v2/"
# checkpoint_path = "./pretrained/checkpoint/inception_v3/model.ckpt"

SavedModel_export_dir = "./pretrained/SavedModel/inception_resnet_v2/"
# SavedModel_export_dir = "./pretrained/SavedModel/inception_v3/"
directory_create(SavedModel_export_dir)
model_version = len(os.listdir(SavedModel_export_dir))


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


def crop_image(image):
    # croped_img = tf.image.resize_image_with_crop_or_pad(image, input_size_h, input_size_w)
    # resized_img = tf.reshape(croped_img, (input_size_h, input_size_w, 3))
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    short_edge = tf.minimum(h, w)
    yy = tf.cast(tf.math.divide(tf.math.subtract(tf.shape(img)[0], short_edge), 2), tf.int32)
    xx = tf.cast(tf.math.divide(tf.math.subtract(tf.shape(img)[1], short_edge), 2), tf.int32)
    crop_img = image[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = tf.image.resize_images(crop_img, (input_size_h, input_size_w))
    resized_img = tf.reshape(resized_img, (input_size_h, input_size_w, 3))
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


def serving_image_decode(image_input):
    image_string, input_image_size = image_input

    # Transform bitstring to uint8 tensor
    image_string = tf.decode_base64(image_string)  # tf-serving will do this automatically
    decoded_image = tf.image.decode_jpeg(image_string, dct_method='INTEGER_ACCURATE')

    reshape_input_tensor = tf.reshape(decoded_image, input_image_size)

    # rgb to bgr
    bgr_input_tensor = tf.reverse(reshape_input_tensor, axis=[-1])

    # Convert to float32 tensor
    input_tensor = tf.cast(bgr_input_tensor, dtype=tf.float32)

    # rescale to interval of [0, 1]
    img_content = tf.math.divide(input_tensor, tf.constant(255.))

    img_input_placeholder = crop_image(img_content)

    return img_input_placeholder


def image_web_saved_encode(image):
    input_bytes = cv2.imencode(".jpg", image)[1].tostring()
    input_image = base64.urlsafe_b64encode(input_bytes)
    image_content = input_image.decode("utf-8")
    return image_content


def image_web_saved_decode(image_str):
    image_bytes = base64.urlsafe_b64decode(image_str)
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def image_encode(image):
    # image -> bytes -> string
    input_bytes = cv2.imencode('.jpg', image)[1].tostring()
    input_image = base64.b64encode(input_bytes)
    image_content = input_image.decode("utf-8")
    return image_content


def image_decode(image_str):
    image_bytes = base64.b64decode(image_str)
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def serving_request(image_content):
    return {"image_bytes": {"b64": image_content}}


if __name__ == "__main__":
    # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    # img = load_image("data/dog/img1.jpg")
    img = cv2.imread("data/dog/img1.jpg")
    # img_str = image_encode(img)
    img_str = image_web_saved_encode(img)
    img_shape = img.shape

    # cv2.imwrite("test.jpg", image_web_saved_decode(img_str))
    # cv2.imwrite("test.jpg", image_decode(img_str))

    # batch_input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    image_string_list = tf.placeholder(tf.string, shape=[None, ], name='image_strings')
    image_shape_list = tf.placeholder(dtype=tf.int32, shape=[None, 3], name="image_shapes")
    batch_input_tensor = tf.map_fn(serving_image_decode, (image_string_list, image_shape_list), dtype=tf.float32)

    # inception_v3
    """
    network = importlib.import_module('models.inception_v3')
    with slim.arg_scope(inception_v3_arg_scope()):
        prelogits, end_points = inception_v3(
                    batch_input_tensor, is_training=False, dropout_keep_prob=1.0,
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
            batch_input_tensor, is_training=False, dropout_keep_prob=1.0,
            num_classes=1001, reuse=None)

    probs = tf.nn.softmax(prelogits, name="probs")
    # """

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # maybe necessary
    tfconfig.allow_soft_placement = True  # maybe necessary
    sess = tf.InteractiveSession(config=tfconfig)
    # sess = tf.InteractiveSession()

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # model restore
    saver = tf.train.Saver()
    saver.restore(sess, pretrained_model_path)
    print("Model Restored")

    # forward pass
    """
    start_time = time.time()
    prob = sess.run(probs, feed_dict={
        image_string_list: [img_str],
        image_shape_list: [img_shape]
    })
    print("End time : %.5ss" % (time.time() - start_time))
    print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing
    # """

    # save as SavedModel
    # """
    # tf serving configure
    # define parameters of output model
    # define output path
    export_path_base = SavedModel_export_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Define input tensor info, the definition of input_images is necessary
    img_tensor_info_input_bytes = tf.saved_model.utils.build_tensor_info(image_string_list)
    img_shape_tensor_info_input_bytes = tf.saved_model.utils.build_tensor_info(image_shape_list)

    # Define output tensor info, the definition of output_result is necessary
    probs_tensor_info_output = tf.saved_model.utils.build_tensor_info(probs)

    # Create signature
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                "image_bytes": img_tensor_info_input_bytes,
                "image_shapes": img_shape_tensor_info_input_bytes
            },
            outputs={
                "probs": probs_tensor_info_output,
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.SERVING],
        signature_def_map={
            "googlenet": prediction_signature})

    # export model
    builder.save(as_text=True)
    print('Done exporting!')
    # """

    # load SavedModel
    # """
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # maybe necessary
    tfconfig.allow_soft_placement = True  # maybe necessary
    with tf.Session(graph=tf.Graph(), config=tfconfig) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.SERVING],
                                   os.path.join(SavedModel_export_dir, max(os.listdir(SavedModel_export_dir))))
        prob = sess.run("probs:0", {
            "image_strings:0": [img_str],
            "image_shapes:0": [img_shape]
        })
        print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing
    # """
