import numpy as np


def concatenate_features(feature, keys):
    concat = np.empty((0, 1), dtype=np.float32)
    for k in feature.keys():
        x = np.reshape(feature['k'] (-1, 1))
        concat = np.concatenate([concat, x])

    return concat

def gen_adjacency_matrix(node_count):
    A = np.ones((node_count, node_count)) - np.eye(node_count)
    return A

def calculate_available_node(x):
    node_idx = np.where(x==0)[0]

    return node_idx
