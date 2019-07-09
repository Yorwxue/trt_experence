# Import TensorFlow and TensorRT
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import skimage
import skimage.io
import skimage.transform


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


# image size for google-net
input_size_h = 299
input_size_w = 299

# test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
img1 = load_image("data/dog/img1.jpg")
img1 = img1.reshape((1, input_size_h, input_size_w, 3))

SavedModel_dir = "./pretrained/SavedModel/inception_resnet_v2/"
# SavedModel_dir = "./pretrained/SavedModel/inception_v3/"
model_tag = "serve"  # can be queried by saved_model_cli
SavedModel_path = os.path.join(
    SavedModel_dir,
    max(os.listdir(SavedModel_dir))
)
print("model path: ", SavedModel_path)
trt_model_dir = "./pretrained/trt/inception_resnet_v2/"
# trt_model_dir = "./pretrained/trt/inception_v3/"

frozen_model_dir = "./pretrained/frozen_model/inception_resnet_v2/"

batch_size = 2
max_GPU_mem_size_for_TRT = 2 << 20

# Inference with TF-TRT `SavedModel` workflow:
# """
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

