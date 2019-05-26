import tensorflow as tf


class FashionMnistArchitectureProvider:

    def encoder_builder(self, x, z_dim):
        h = x
        h = tf.reshape(h, [-1, 28, 28, 1])
        print('Creating encoder')
        for filters_count in [128, 256, 512, 1024]:
            h = tf.layers.conv2d(h, kernel_size=(4, 4), strides=(2, 2), padding="SAME", filters=filters_count,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.0099999))
            h = tf.nn.relu(h)

        h = tf.layers.flatten(h, name='Encoder_Flatten')
        return tf.layers.dense(h, z_dim, name=f'Encoder_output')

    def decoder_builder(self, z, x_dim):
        h = tf.layers.dense(z, units=7*7*1024)
        h = tf.reshape(h, [-1, 7, 7, 1024])

        for filters_count in [512, 256]:
            h = tf.layers.conv2d_transpose(h, kernel_size=(4, 4), strides=(2, 2), padding="SAME", filters=filters_count,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.0099999))
            h = tf.nn.relu(h)

        h = tf.layers.conv2d_transpose(h, kernel_size=(4, 4), padding="SAME", activation=tf.nn.tanh, filters=1)

        h = tf.reshape(h, [-1, 28, 28])
        h = tf.div(h, 2.0) + 0.5

        return h
