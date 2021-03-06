import tensorflow as tf
import numpy as np
from typing import Tuple
from rec_errors import sqrt_mean_squared_euclidean_norm


class AutoEncoderModel:

    def __init__(self, x_dim, z_dim: int, iterator: tf.data.Iterator, normality_test):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.__iterator = iterator

        if normality_test is None:
            raise ValueError('normality_test cannot be null')
        self.__normality_test = normality_test

    def build_architecture(self,
                           encoder_builder,  # should return tensor Z - encoded(X)
                           decoder_builder,  # should return decoded(Z)
                           ):
        self.tensor_input_x = self.__iterator.get_next()
        self.tensor_z = encoder_builder(self.tensor_input_x, self.z_dim)
        self.tensor_output_x = decoder_builder(self.tensor_z, self.x_dim)

    def build_cost_tensor(self, tensor_extra_latent: tf.Tensor):
        self.tensor_rec_error = sqrt_mean_squared_euclidean_norm(self.tensor_input_x, self.tensor_output_x)

        tensor_normality_test_z = self.tensor_z
        if tensor_extra_latent is not None:
            print('Adding tensor with extra latent.')
            tensor_normality_test_z = tf.concat([self.tensor_z, tensor_extra_latent], 0)

        self.tensor_normality_cost = self.__normality_test(tensor_normality_test_z)

        return self.tensor_rec_error + self.tensor_normality_cost

    def encode(self, sess: tf.Session, x: list):
        return sess.run(self.tensor_z, feed_dict={self.tensor_input_x: x})

    def decode(self, sess: tf.Session, z: list):
        return sess.run(self.tensor_output_x, feed_dict={self.tensor_z: z})

    def encode_decode(self, sess: tf.Session, x: list) -> Tuple[np.ndarray, np.ndarray]:
        return sess.run([self.tensor_z, self.tensor_output_x],
                        feed_dict={self.tensor_input_x: x})
