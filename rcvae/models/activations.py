import tensorflow as tf
from keras import backend as K


def mean_activation(x):
    return tf.clip_by_value(K.exp(x), 1e-5, 1e6)


def disp_activation(x):
    return tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
