import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import pybase64 as base64
import matplotlib.pyplot as plt


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


saved_model_path = os.path.abspath("./weights/SaveModel")

# Add the following two command to avoid cuDNN failed to initialize
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# def crop_image(image):
#     # croped_img = tf.image.resize_image_with_crop_or_pad(image, input_size_h, input_size_w)
#     # resized_img = tf.reshape(croped_img, (input_size_h, input_size_w, 3))
#     shape = tf.shape(image)
#     h = shape[0]
#     w = shape[1]
#     short_edge = tf.minimum(h, w)
#     yy = tf.cast(tf.math.divide(tf.math.subtract(tf.shape(img)[0], short_edge), 2), tf.int32)
#     xx = tf.cast(tf.math.divide(tf.math.subtract(tf.shape(img)[1], short_edge), 2), tf.int32)
#     crop_img = image[yy: yy + short_edge, xx: xx + short_edge]
#     resized_img = tf.image.resize_images(crop_img, (input_size_h, input_size_w))
#     resized_img = tf.reshape(resized_img, (input_size_h, input_size_w, 3))
#     return resized_img


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
    img_input_placeholder = tf.math.divide(input_tensor, tf.constant(255.))

    # img_input_placeholder = crop_image(img_input_placeholder)

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
    # prepare dataset
    # """
    tfds.disable_progress_bar()

    SPLIT_WEIGHTS = (8, 1, 1)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

    # downloads and caches the data
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs', split=list(splits),
        with_info=True, as_supervised=True)

    print(raw_train)
    print(raw_validation)
    print(raw_test)

    # Show the first two images and labels from the training set:
    get_label_name = metadata.features['label'].int2str

    for image, label in raw_train.take(2):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))

    # Resize the images to a fixes input size, and rescale the input channels to a range of [-1,1]
    IMG_SIZE = 160  # All images will be resized to 160x160
    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    # shuffle and batch the data.
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    # Inspect a batch of data:
    for image_batch, label_batch in train_batches.take(1):
        pass
    print("image_batch.shape: ", image_batch.shape)
    # """

    # load model
    base_learning_rate = 0.0001
    model = tf.keras.experimental.load_from_saved_model(saved_model_path)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # forward pass
    # """
    validation_steps = 20
    loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    # """

    # # save as SavedModel
    # # """
    # # tf serving configure
    # # define parameters of output model
    # # define output path
    # export_path_base = SavedModel_export_dir
    # export_path = os.path.join(
    #     tf.compat.as_bytes(export_path_base),
    #     tf.compat.as_bytes(str(model_version)))
    # print('Exporting trained model to', export_path)
    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    #
    # # Define input tensor info, the definition of input_images is necessary
    # img_tensor_info_input_bytes = tf.saved_model.utils.build_tensor_info(image_string_list)
    # img_shape_tensor_info_input_bytes = tf.saved_model.utils.build_tensor_info(image_shape_list)
    #
    # # Define output tensor info, the definition of output_result is necessary
    # probs_tensor_info_output = tf.saved_model.utils.build_tensor_info(probs)
    #
    # # Create signature
    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={
    #             "image_bytes": img_tensor_info_input_bytes,
    #             "image_shapes": img_shape_tensor_info_input_bytes
    #         },
    #         outputs={
    #             "probs": probs_tensor_info_output,
    #         },
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    #
    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.SERVING],
    #     signature_def_map={
    #         "googlenet": prediction_signature})
    #
    # # export model
    # builder.save(as_text=True)
    # print('Done exporting!')
    # # """
    #
    # # load SavedModel
    # # """
    # tfconfig = tf.ConfigProto()
    # tfconfig.gpu_options.allow_growth = True  # maybe necessary
    # tfconfig.allow_soft_placement = True  # maybe necessary
    # with tf.Session(graph=tf.Graph(), config=tfconfig) as sess:
    #     tf.saved_model.loader.load(sess, [tf.saved_model.SERVING],
    #                                os.path.join(SavedModel_export_dir, max(os.listdir(SavedModel_export_dir))))
    #     prob = sess.run("probs:0", {
    #         "image_strings:0": [img_str],
    #         "image_shapes:0": [img_shape]
    #     })
    #     print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing
    # # """
