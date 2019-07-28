import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import onnxruntime as rt
import time


onnx_model_dir = os.path.abspath(os.path.join(__file__, "..", "onnx_model", "cnn_model"))
onnx_model_dir = os.path.join(onnx_model_dir, str(len(os.listdir(onnx_model_dir))-2))
onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")

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
# x_test = [image_web_saved_encode(np.concatenate([image, image, image], axis=2) * 255) for image in list(x_test)]

# fill 3 channels with copy
x_test = [np.concatenate([image, image, image], axis=2) for image in list(x_test)]
# """

sess = rt.InferenceSession(onnx_model_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

START_TIME = time.time()
pred_onx = sess.run([label_name], {
    input_name: np.asarray([x_test[0]])
})[0]
print("spent %f seconds" % (time.time() - START_TIME))
print("label: %d, prediction: %d" % (np.argmax(y_test[0]), np.argmax(pred_onx[0])))
