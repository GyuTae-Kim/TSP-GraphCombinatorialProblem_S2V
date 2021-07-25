import tensorflow as tf
import numpy as np
from scipy.spatial import distance_matrix


def gen_init_nodefeat(coord):
    x = np.ones((coord.shape[0], 6), dtype=np.float32)
    x[:, :2] = coord
    x[0, 2:5] = 0.      # Start Node
    return x

def gen_init_edgefeat(weights):
    N = weights.shape[0]
    di = np.diag_indices(N, ndim=2)
    w = np.zeros((N, N, 4), dtype=np.float32)
    w[di[0], di[1], 0] = 1.
    w[0, 1:, 0] = 1.
    w[:, :, 1] = weights
    w[di[0], di[1], 2] = 1.
    w[0, 1:, 2] = 1.
    w[1:, 0, 2] = 1.
    w[:, :, 3] = 1.
    return w

def gen_adjacency_matrix(node_count):
    A = np.ones((node_count, node_count), dtype=np.float32) - np.eye(node_count, dtype=np.float32)
    return A

def calculate_x(x, next_node):
    x[next_node, 0] = 1.
    return x

def check_done(x):
    s = np.sum(x)
    if s == 0:
        return True
    else:
        return False

def calculate_weights(feature):
    weights = distance_matrix(feature, feature)
    return weights
