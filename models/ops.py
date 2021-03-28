import numpy as np
import tensorflow as tf


def specific_value(mu, idx, length):
    h = np.zeros(length)
    h[idx] = 1.
    h = tf.convert_to_tensor(h,
                             dtype=tf.float32)
    s_val = mu * h
    s_val = tf.reduce_sum(s_val, axis=1, keepdims=True)

    return s_val
