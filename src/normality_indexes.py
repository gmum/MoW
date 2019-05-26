import tensorflow as tf
import numpy as np
from rec_errors import euclidean_norm_squared


# CWAE
def silverman_rule_of_thumb(N: int):
    return tf.pow(4/(3*N), 0.4)


def cw(X):
    D = tf.cast(tf.shape(X)[1], tf.float32)
    N = tf.cast(tf.shape(X)[0], tf.float32)
    y = silverman_rule_of_thumb(N)

    K = 1/(2*D-3)

    A1 = euclidean_norm_squared(tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1)), axis=2)
    A = (1/(N**2)) * tf.reduce_sum((1/tf.sqrt(y + K*A1)))

    B1 = euclidean_norm_squared(X, axis=1)
    B = (2/N)*tf.reduce_sum((1/tf.sqrt(y + 0.5 + K*B1)))

    return (1/tf.sqrt(1+y)) + A - B


def log_cw(X):
    return tf.log(cw(X))


# WAE-MMD
def mmd_penalty(sample_qz, sample_pz):
    n = tf.cast(tf.shape(sample_qz)[0], tf.float32)
    d = tf.cast(tf.shape(sample_qz)[1], tf.float32)

    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)

    sigma2_p = 1. ** 2
    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * tf.matmul(sample_pz, sample_pz, transpose_b=True)

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * tf.matmul(sample_qz, sample_qz, transpose_b=True)

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    Cbase = 2. * d * sigma2_p

    stat = 0.
    TempSubtract = 1. - tf.eye(n)
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + distances_qz) + C / (C + distances_pz)
        res1 = tf.multiply(res1, TempSubtract)
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = C / (C + distances)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat += res1 - res2
    return stat


def wae_normality_index(Z: tf.Tensor, z_dim: int):
    dist = tf.distributions.Normal(np.zeros(z_dim, dtype=np.float32), np.ones(z_dim, dtype=np.float32))
    tensor_input_latent_sample = dist.sample(tf.shape(Z)[0])
    return mmd_penalty(Z, tensor_input_latent_sample)


# SWAE
def swae_normality_index_inner(projected_latent, theta, z_dim):
    n = tf.cast(tf.shape(projected_latent)[0], tf.int32)

    dist = tf.distributions.Normal(np.zeros(z_dim, dtype=np.float32), np.ones(z_dim, dtype=np.float32))
    sample = dist.sample(n)

    projz = tf.keras.backend.dot(sample, tf.transpose(theta))
    transposed_projected_latent = tf.transpose(projected_latent)
    transpose_projected_sample = tf.transpose(projz)

    W2 = (tf.nn.top_k(transposed_projected_latent, k=n).values -
          tf.nn.top_k(transpose_projected_sample, k=n).values)**2

    return W2


def swae_normality_index(Z: tf.Tensor, z_dim: int):
    randomed_normal = tf.random_normal(shape=(50, z_dim))
    theta = randomed_normal / tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(randomed_normal), axis=1)), (-1, 1))
    projae = tf.keras.backend.dot(Z, tf.transpose(theta))
    normality_test_result = swae_normality_index_inner(projae, theta, z_dim)
    return tf.reduce_mean(normality_test_result)
