import numpy as np


def _calculate_with_adjacency(node, A):
    if A[node] == 1.:
        return True
    else:
        return False

def _calculate_weights(node, feature):
    p1_x = feature[node, :1]
    p1_y = feature[node, 1:]
    p2_x = feature[:, :1]
    p2_y = feature[:, 1:]
    weights = np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)
    
    return weights.tolist()

vec_calculate_weight_adjacency = np.vectorize(_calculate_with_adjacency)
vec_calculate_weights = np.vectorize(_calculate_weights,
                                     excluded=['feature'],
                                     otypes=[object])

'''
def calculate_available_node(node_list, x, connected_all=True, A=None):
    _x = np.squeeze(x, axis=-1)
    node_idx = np.where(_x==0)[0]

    if not connected_all:
        if A is None:
            raise ValueError('    [Err] If connected_all == False, adjency matrix and current node must not be None')
        if len(A.shape) != 1:
            raise ValueError('    [Err] Adjency matrix rank must be 1.')
        mask = vec_calculate_weight_adjacency(node_idx, A)
        node_idx = node_idx[mask]
    
    node = node_list[node_idx]

    return node
'''

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

##### If you want to use custom data, fix this part #####
def euclidean_distance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p1[1]) ** 2)
    return dist
#########################################################

def check_done(x):
    if 0. in x:
        return False
    else:
        return True

def calculate_weights(node, feature):
    weights = vec_calculate_weights(node=np.array(node), feature=np.array(feature)).tolist()
    weights = np.array(weights, dtype=np.float32)
    weights = np.squeeze(weights, axis=-1)

    return weights
