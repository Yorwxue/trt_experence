# from checkpoint to SavedModel
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import pybase64 as base64

from googlenet.nets.cnn_model.CNN_MODEL import cnn_model


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


checkpoint_dir = os.path.abspath("./checkpoint/cnn_model")
print("checkpoint path:", checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, "model")


SavedModel_export_dir = "./SavedModel/cnn_model/"
directory_create(SavedModel_export_dir)
summaries_dir = "./SavedModel/cnn_model/tensorboard/"
directory_create(summaries_dir)
new_model_version = len(os.listdir(SavedModel_export_dir)) - 1
input_height = 28
input_width = 28


def serving_image_decode(image_input):
    image_string, input_image_size = image_input

    # Transform bitstring to uint8 tensor
    image_string = tf.decode_base64(image_string)  # tf-serving will do this automatically
    decoded_image = tf.image.decode_jpeg(image_string, dct_method='INTEGER_ACCURATE')

    reshape_input_tensor = tf.reshape(decoded_image, input_image_size)
    reshape_input_tensor = tf.reshape(reshape_input_tensor, (input_height, input_width, 3))

    # rgb to bgr
    bgr_input_tensor = tf.reverse(reshape_input_tensor, axis=[-1])

    # Convert to float32 tensor
    input_tensor = tf.cast(bgr_input_tensor, dtype=tf.float32)

    # rescale to interval of [0, 1]
    img_input_placeholder = tf.math.divide(input_tensor, tf.constant(255.))

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
    x_test = [image_web_saved_encode(np.concatenate([image, image, image], axis=2)*255) for image in list(x_test)]
    # """

    image_string_list = tf.placeholder(tf.string, shape=[None, ], name='image_strings')
    image_shape_list = tf.placeholder(dtype=tf.int32, shape=[None, 3], name="image_shapes")
    batch_input_tensor = tf.map_fn(serving_image_decode, (image_string_list, image_shape_list), dtype=tf.float32)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # maybe necessary
    tfconfig.allow_soft_placement = True  # maybe necessary
    sess = tf.InteractiveSession(config=tfconfig)

    # build model
    model = cnn_model(batch_input_tensor, 10)

    # forward pass
    forward = True
    if forward:
        label_batch = tf.placeholder(tf.float32, shape=(None, 10), name="label_batch")
        model.build_graph(label_batch)

        # initialize
        sess.run(tf.global_variables_initializer())
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        sess.run(running_vars_initializer)

        # model restore
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        print("Model Restored")

        start_time = time.time()
        cost, accuracy = sess.run([model.cost, model.acc_op], feed_dict={
            # image_list: x_test,
            image_string_list: x_test,
            image_shape_list: [(28, 28, 3)]*len(x_test),
            model.labels: y_test
        })
        print("testing loss: ", cost)
        print("testing accuracy: ", accuracy)
        print("End time : %.5ss" % (time.time() - start_time))

    # save as SavedModel
    SavedModel = True
    if SavedModel:
        # tf serving configure
        # define parameters of output model
        # define output path
        export_path_base = SavedModel_export_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(new_model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Define input tensor info, the definition of input_images is necessary
        img_tensor_info_input_bytes = tf.saved_model.utils.build_tensor_info(image_string_list)
        img_shape_tensor_info_input_bytes = tf.saved_model.utils.build_tensor_info(image_shape_list)

        # Define output tensor info, the definition of output_result is necessary
        logits_tensor_info_output = tf.saved_model.utils.build_tensor_info(model.logits)

        # Create signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    "image_bytes": img_tensor_info_input_bytes,
                    "image_shapes": img_shape_tensor_info_input_bytes
                },
                outputs={
                    "logits": logits_tensor_info_output,
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.SERVING],
            signature_def_map={
                "cnn_model": prediction_signature})

        # export model
        builder.save(as_text=True)
        print('Done exporting!')

    # load SavedModel
    # """
    test_idx = 0
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # maybe necessary
    tfconfig.allow_soft_placement = True  # maybe necessary
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(graph=tf.Graph(), config=tfconfig) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.SERVING],
                                   os.path.join(SavedModel_export_dir,  str(len(os.listdir(SavedModel_export_dir))-2)))
        # warm up
        print("warm up")
        for i in range(5):
            prob = sess.run("logits:0", {
                "image_strings:0": [x_test[test_idx]],
                "image_shapes:0": [(28, 28, 3)]
            })
        print("counter start")
        START_TIME = time.time()
        prob = sess.run("logits:0", {
            "image_strings:0": [x_test[test_idx]],
            "image_shapes:0": [(28, 28, 3)]
        })
        print("label: %d, prediction: %d" % (np.argmax(y_test[test_idx]), np.argmax(prob[0])))
        print("spent %f seconds" % (time.time()-START_TIME))
    # """

    # write graph
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir, tf.get_default_graph())
