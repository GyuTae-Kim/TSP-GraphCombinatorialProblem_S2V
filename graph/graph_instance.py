import numpy as np

import copy

import graph.ops as ops


class Instance(object):

    def __init__(self, n_city, feature, node_feat_size, edge_feat_size):
        if not (n_city == feature.shape[0]):
            raise ValueError('   [Err] Parameters\' length are not match.'
                             'city_list: {}, feature: {}'.format(n_city,
                                                                 feature.shape))

        self.n_city = n_city
        self.coord = feature
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size

        self.node_list = np.arange(n_city, dtype=np.int32)
        self.available_node = np.arange(1, n_city, dtype=np.int32).tolist()
        self.A = ops.gen_adjacency_matrix(n_city)

        self.current_node = 0
        self.weight = ops.calculate_weights(self.coord)
        self.step_count = 0
        self.path = [self.current_node]
        self.node_feat = ops.gen_init_nodefeat(self.coord)
        self.edge_feat = ops.gen_init_edgefeat(self.weight)
        self.di = np.diag_indices(n_city, ndim=2)
        self.cov = np.zeros((n_city,), dtype=np.int8)

    def instance_info(self):
        info = {'node_list': self.node_list,
                'adj': self.A,
                'feature': self.coord}
        
        return info

    def move(self, next_node, idx=None):
        if next_node not in self.available_node:
            raise ValueError('   [Err] Unable to move to that node.'
                             'now: {} dest: {}'.format(self.current_node,
                                                       next_node))
        
        self.node_feat[next_node, 4] = 0.
        self.edge_feat[next_node, :, 0] = 1.
        ch2 = (self.node_feat[:, 4].astype(np.int8) ^ self.cov).astype(np.float32)
        ch2[next_node] = 1.
        self.edge_feat[next_node, :, 2] = ch2
        self.edge_feat[:, next_node, 2] = ch2
        
        done = ops.check_done(self.node_feat[:, 4])
        self.current_node = next_node
        self.step_count += 1

        if idx is None:
            self.path.append(self.current_node)
        else:
            self.path.insert(idx, self.current_node)
        
        self.available_node.remove(next_node)
        
        if len(self.available_node) == 0:
            done = True
        
        return done

    def __len__(self):
        return self.n_city
    
    def get_edgefeat(self):
        return copy.deepcopy(self.edge_feat)

    def get_available_node(self):
        return copy.deepcopy(self.available_node)
    
    def get_adjacency_matrix(self):
        return self.A
    
    def get_nodefeat(self):
        return copy.deepcopy(self.node_feat)

    def get_path(self):
        return copy.deepcopy(self.path)

    def get_coord(self):
        return self.coord
    
    def get_weight(self):
        return self.weight
