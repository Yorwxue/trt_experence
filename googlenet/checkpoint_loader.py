import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


saved_model_path = os.path.abspath("./weights/SaveModel")
print(saved_model_path)
directory_create(saved_model_path)

# Add the following two command to avoid cuDNN failed to initialize
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == "__main__":
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

    # Create the base model from the pre-trained convnets
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # feature extractor
    feature_batch = base_model(image_batch)
    print("feature_batch.shape: ", feature_batch.shape)

    # Freeze the convolutional base
    base_model.trainable = False

    # show architecture of the base model
    base_model.summary()

    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print("feature_batch_average.shape", feature_batch_average.shape)

    #  convert these features into a single prediction per image
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print("prediction_batch.shape", prediction_batch.shape)

    # stack the feature extracto
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    # compile model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    print("trainable variables: ", len(model.trainable_variables))

    # train
    num_train, num_val, num_test = (
        metadata.splits['train'].num_examples*weight/10
        for weight in SPLIT_WEIGHTS
    )
    initial_epochs = 10
    steps_per_epoch = round(num_train)//BATCH_SIZE
    validation_steps = 20

    loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches)
    predictions = model.predict(test_batches)

    # Learning curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


    # Export the model to a SavedModel
    tf.keras.experimental.export_saved_model(model, saved_model_path)

    # Recreate the exact same model
    # new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)

    # Check that the state is preserved
    # new_predictions = new_model.predict(test_batches)
    # np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)
