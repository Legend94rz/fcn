import tensorflow as tf
import numpy as np


def conv2d(x, filter_shape, name, activation=None, use_bias=True, kernel_initializer=tf.initializers.truncated_normal,
           bias_initializer=tf.initializers.zeros, **kwargs):
    r"""
    :param x: input Tensor.
    :param filter_shape: `[filter_height, filter_width, in_channels, out_channels]`
    :param name: a custom name.
    :param activation: function.
    :param use_bias: boolean.
    :param kernel_initializer: function
    :param bias_initializer: boolean.
    :param kwargs: another params passed to tf.nn.conv2d. strides and padding must be assigned.
    :return: output Tensor
    """
    with tf.variable_scope(name):
        filter_shape = filter_shape if callable(kernel_initializer) else None
        w = tf.get_variable("w", shape=filter_shape, initializer=kernel_initializer)
        filter_shape = w.shape
        if use_bias:
            bias_shape = (filter_shape[-1],) if callable(bias_initializer) else None
            b = tf.get_variable(name="b", shape=bias_shape, initializer=bias_initializer)
            middle = tf.nn.conv2d(x, w, **kwargs) + b
        else:
            middle = tf.nn.conv2d(x, w, **kwargs)
    if activation is None:
        return middle
    assert (callable(activation))
    return activation(middle)


def conv2d_transpose(x, filter_shape, name, activation=None, use_bias=True,
                     kernel_initializer=tf.initializers.truncated_normal,
                     bias_initializer=tf.initializers.zeros, **kwargs):
    r"""
    :param x: input Tensor.
    :param filter_shape: `[height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    :param name:
    :param activation:
    :param use_bias:
    :param kernel_initializer:
    :param bias_initializer:
    :param kwargs: padding, strides, output_shape
    :return:
    """
    with tf.variable_scope(name):
        filter_shape = filter_shape if callable(kernel_initializer) else None
        w = tf.get_variable("w", shape=filter_shape, initializer=kernel_initializer)
        filter_shape = w.shape
        if use_bias:
            bias_shape = (filter_shape[-2],) if callable(bias_initializer) else None
            b = tf.get_variable(name="b", shape=bias_shape, initializer=bias_initializer)
            middle = tf.nn.conv2d_transpose(x, w, **kwargs) + b
        else:
            middle = tf.nn.conv2d_transpose(x, w, **kwargs)
    if activation is None:
        return middle
    assert (callable(activation))
    return activation(middle)


def flatten(x):
    return tf.reshape(x, tf.stack([-1, np.prod([x.value for x in x.shape[1:]])]))


def fully_connected(x, name, output_size, activation=None, use_bias=True,
                    kernel_initializer=tf.initializers.glorot_uniform, bias_initializer=tf.initializers.zeros):
    input_size = x.shape[1].value
    kernel_shape = (input_size, output_size) if callable(kernel_initializer) else None
    with tf.variable_scope(name):
        w = tf.get_variable(name="w", shape=kernel_shape, initializer=kernel_initializer)
        if use_bias:
            bias_shape = (output_size,) if callable(bias_initializer) else None
            b = tf.get_variable(name="b", shape=bias_shape, initializer=bias_initializer)
            middle = tf.matmul(x, w) + b
        else:
            middle = tf.matmul(x, w)
    if activation is None:
        return middle
    assert (callable(activation))
    return activation(middle)
