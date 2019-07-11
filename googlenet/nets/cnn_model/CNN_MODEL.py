import tensorflow as tf


class cnn_model(object):
    def __init__(self, input_placeholder, num_classes):
        self.inputs = input_placeholder
        self.num_classes = num_classes
        self._build_model()
        self.labels = None

    def build_training_graph(self, labels):
        self.labels = labels
        self._build_train_op()
        return

    def _build_model(self):
        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                x = self._conv2d(self.inputs, name='cnn-1', filter_size=3, in_channels=3, out_channels=63, strides=1)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides=2)

            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, name='cnn-2', filter_size=3, in_channels=64, out_channels=128, strides=1)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides=2)

            with tf.variable_scope('unit-3'):
                x = self._conv2d(x, name='cnn-3', filter_size=3, in_channels=64, out_channels=128, strides=1)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides=2)

            with tf.variable_scope('unit-4'):
                x = self._conv2d(x, name='cnn-4', filter_size=3, in_channels=128, out_channels=256, strides=1)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides=2)

        with tf.variable_scope('fc'):
            # [batch_size, max_stepsize, num_features]
            batch_size, height, width, channels = x.get_shape().as_list()
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.reshape(x, [-1, width, height * channels])

            outputs = x
            outputs = tf.reshape(outputs, [-1, 64])

            W = tf.get_variable(name='w',
                                shape=[64, self.num_classes],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',
                                shape=[self.num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, self.num_classes])
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
        )

        self.optimizer = tf.train.AdamOptimizer().minimize(
            self.loss,
            global_step=self.global_step
        )
        train_ops = [self.optimizer]
        self.train_op = tf.group(*train_ops)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='conv',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(name='bais',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
