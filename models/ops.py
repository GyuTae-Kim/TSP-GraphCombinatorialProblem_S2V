import numpy as np
import tensorflow as tf


def specific_value(mu, idx):
    len_idx = len(idx)
    brod = tf.ones((len_idx, *mu.shape), dtype=tf.float32)
    brod_mu = tf.expand_dims(mu, axis=0) * brod
    h = np.zeros_like(brod_mu, dtype=np.float32)
    h[np.arange(len_idx), idx, :] = 1.
    h = tf.convert_to_tensor(h, dtype=tf.float32)
    s_val = brod_mu * h
    s_val = tf.reduce_sum(s_val, axis=1)

    return s_val
