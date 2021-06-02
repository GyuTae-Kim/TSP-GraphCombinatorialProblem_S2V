import numpy as np

import copy

import graph.ops as ops


class Instance(object):

    def __init__(self, n_city, feature):
        if not (n_city == feature.shape[0]):
            raise ValueError('   [Err] Parameters\' length are not match.'
                             'city_list: {}, feature: {}'.format(n_city,
                                                                 feature.shape))

        self.n_city = n_city
        self.feature = feature

        self.node_list = np.arange(n_city, dtype=np.int32)
        self.available_node = np.arange(1, n_city, dtype=np.int32).tolist()
        self.A = ops.gen_adjacency_matrix(n_city)
        self.x = ops.gen_init_x(n_city)
        self.cost_func = ops.euclidean_distance

        self.current_node = 0
        self.weight = ops.calculate_weights(self.node_list,
                                            self.feature)
        self.step_count = 0
        self.path = [self.current_node]

    def instance_info(self):
        info = {'node_list': self.node_list,
                'adj': self.A,
                'feature': self.feature}
        
        return info

    def move(self, next_node, idx=None):
        if next_node not in self.available_node:
            raise ValueError('   [Err] Unable to move to that node.'
                             'now: {} dest: {}'.format(self.current_node,
                                                       next_node))
        
        self.x = ops.calculate_x(self.x, next_node)

        done = ops.check_done(self.x)

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
    
    def get_weight(self):
        return self.weight

    def get_available_node(self):
        return self.available_node
    
    def get_adjacency_matrix(self):
        return self.A
    
    def get_x(self):
        return copy.deepcopy(self.x)

    def get_path(self):
        return self.path

    def get_feature(self):
        return self.feature
