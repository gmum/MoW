import tensorflow as tf


def euclidean_norm_squared(X, axis=None):
    return tf.reduce_sum(tf.square(X), axis=axis)


def squared_euclidean_norm(input, output):
    return euclidean_norm_squared(input - output, axis=1)


def mean_squared_euclidean_norm(x, y):
    return tf.reduce_mean(squared_euclidean_norm(tf.layers.flatten(x), tf.layers.flatten(y)))


def sqrt_mean_squared_euclidean_norm(x, y):
    return tf.sqrt(mean_squared_euclidean_norm(x, y))
