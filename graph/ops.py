import numpy as np


def calculate_available_node(node_list, x, connected_all=True, A=None):
    node_idx = np.where(x==0)[0]

    if not connected_all:
        if A is None:
            raise ValueError('    [Err] If connected_all == False, adjency matrix and current node must not be None')
        if len(A.shape) != 1:
            raise ValueError('    [Err] Adjency matrix rank must be 1.')
        vec_cal = np.vectorize(_vec_caculate_with_adjacency)
        mask = vec_cal(node_idx, A)
        node_idx = node_idx[mask]
    
    node = node_list[node_idx]

    return node

def _vec_caculate_with_adjacency(node, A):
    if A[node] == 1.:
        return True
    else:
        return False

def gen_init_x(node_count):
    x = np.zeros(node_count)
    x[0] = 1.
    return x

def gen_adjacency_matrix(node_count):
    A = np.ones((node_count, node_count)) - np.eye(node_count)
    return A

def calculate_x(x, next_node):
    x[next_node] = 1.
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
    vec_calc_dist = np.vectorize(euclidean_distance)
    weights = vec_calc_dist(feature[node], feature)
    return weights