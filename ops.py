import numpy as np


def concatenate_features(feature):
    concat = None
    for k in feature.keys():
        x = np.reshape(feature[k], (-1, 1))
        if concat is None:
            concat = x
        else:
            concat = np.concatenate([concat, x], axis=1)

    return concat

def gen_adjacency_matrix(node_count):
    A = np.ones((node_count, node_count)) - np.eye(node_count)
    return A

def calculate_available_node(x):
    node_idx = np.where(x[:, 3] == 0)[0]

    return node_idx
