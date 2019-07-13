import time
import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from googlenet.nets.cnn_model.CNN_MODEL import cnn_model


def directory_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    weights_dir = os.path.abspath("./weights/cnn_model")
    directory_create(weights_dir)
    model_name = "model"

    checkpoint_dir = os.path.abspath("./checkpoint/cnn_model")
    directory_create(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, model_name)

    # preparing dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    # reshape from 784 to 28*28
    x_train = np.reshape(x_train, [x_train.shape[0], 28, 28, 1])
    x_test = np.reshape(x_test, [x_test.shape[0], 28, 28, 1])

    # fill 3 channel with copy
    x_train = [np.concatenate([image]*3, axis=2) for image in list(x_train)]
    x_test = [np.concatenate([image]*3, axis=2) for image in list(x_test)]

    image_batch = tf.placeholder(tf.float32, shape=(None, 28, 28, 3), name="image_batch")

    # build model
    model = cnn_model(image_batch, 10)

    # train
    TRAIN = True
    if TRAIN:
        num_epochs = 10
        batch_size = 100
        save_steps = 1000
        validation_steps = 100
        num_batches_per_epoch = int(len(x_train)/batch_size)
        pretrained = False

        label_batch = tf.placeholder(tf.float32, shape=(None, 10), name="label_batch")
        model.build_graph(label_batch)

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True  # maybe necessary
        tfconfig.allow_soft_placement = True  # maybe necessary
        with tf.Session(config=tfconfig) as sess:
            # initial
            sess.run(tf.global_variables_initializer())
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            sess.run(running_vars_initializer)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            if pretrained:
                ckpt = tf.train.latest_checkpoint(weights_dir)
                if ckpt:
                    # the global_step will restore sa well
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            for cur_epoch in range(num_epochs):
                start_time = time.time()
                batch_time = time.time()

                # the tracing part
                for batch_idx in range(num_batches_per_epoch):
                    if (batch_idx + 1) % 100 == 0:
                        print('batch', batch_idx, ': time', time.time() - batch_time)
                    batch_time = time.time()
                    batch_inputs = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    batch_labels = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels}

                    batch_loss, step, _ = \
                        sess.run([model.cost, model.global_step, model.train_op], feed)

                    # save the checkpoint
                    if step % save_steps == 1:
                        if not os.path.isdir(weights_dir):
                            os.mkdir(weights_dir)
                        saver.save(sess, os.path.join(weights_dir, model_name),
                                   global_step=step)

                    # do validation
                    if step % validation_steps == 0:
                        val_feed = {model.inputs: x_test,
                                    model.labels: y_test}

                        cost, accuracy = sess.run([model.cost, model.acc_op], val_feed)

                        # train_err /= num_train_samples
                        now = datetime.datetime.now()
                        log = "Epoch {}/{}, " \
                              "accuracy = {:.5f},train_cost = {:.5f}, " \
                              ", time = {:.3f}"
                        print(log.format(cur_epoch + 1, num_epochs, accuracy, batch_loss,
                                         time.time() - start_time))

            # save as checkpoint
            checkpoint = True
            if checkpoint:
                saver = tf.train.Saver()
                saver.save(sess, checkpoint_path)
                print("Ckeckpoint Saved at: %s" % checkpoint_dir)

    print("Start Evaluate")
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # maybe necessary
    tfconfig.allow_soft_placement = True  # maybe necessary
    with tf.Session(config=tfconfig) as sess:
        # forward pass
        label_batch = tf.placeholder(tf.float32, shape=(None, 10), name="label_batch")
        model.build_graph(label_batch)

        # initialize
        sess.run(tf.global_variables_initializer())
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        sess.run(running_vars_initializer)

        # model restore
        saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state(weights_dir)
        # saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, checkpoint_path)
        print("Model Restored")

        start_time = time.time()
        cost, accuracy = sess.run([model.cost, model.acc_op], feed_dict={
            model.inputs: x_test,
            model.labels: y_test
        })
        print("testing loss: ", cost)
        print("testing accuracy: ", accuracy)
        print("End time : %.5ss" % (time.time() - start_time))

        # single data evaluate
        prob = sess.run(model.logits, {
            model.inputs: [x_test[0]],
        })
        print("label: %d, prediction: %d" % (np.argmax(y_test[0]), np.argmax(prob[0])))
