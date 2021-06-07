import numpy as np
from scipy.spatial import distance_matrix


def gen_init_x(node_count):
    x = np.zeros((node_count, 1), dtype=np.float32)
    x[0, 0] = 1.
    return x

def gen_adjacency_matrix(node_count):
    A = np.ones((node_count, node_count), dtype=np.float32) - np.eye(node_count, dtype=np.float32)
    return A

def calculate_x(x, next_node):
    x[next_node, 0] = 1.
    return x

def check_done(x):
    if 0. in x:
        return False
    else:
        return True

def calculate_weights(feature):
    weights = distance_matrix(feature, feature)

    return weights
