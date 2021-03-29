import numpy as np

import graph.ops as ops


class Instance(object):

    def __init__(self, n_city, feature):
        if not (n_city == feature.shape[0]):
            raise ValueError('   [Err] Parameters\' length are not match.'
                             'city_list: {}, feature: {}'.format(n_city,
                                                                 feature.shape))

        self.n_city = n_city
        self.feature = feature

        self.node_list = np.arange(n_city)
        self.A = ops.gen_adjacency_matrix(n_city)
        self.x = ops.gen_init_x(n_city)
        self.cost_func = ops.euclidean_distance

        self.current_node = 0
        self.available_node = ops.calculate_available_node(self.node_list,
                                                           self.x)
        self.weight = ops.calculate_weights(self.current_node,
                                                    self.feature)
        self.total_cost = 0.
        self.step_count = 0
        self.path = [self.current_node]

    def instance_info(self):
        info = {'node_list': self.node_list,
                'adj': self.A,
                'feature': self.feature}
        
        return info

    def move(self, next_node):
        if next_node not in self.available_node:
            raise ValueError('   [Err] Unable to move to that node.'
                             'now: {} dest: {}'.format(self.current_node,
                                                       next_node))
        
        self.x = ops.calculate_x(self.x, next_node)
        cost = ops.euclidean_distance(self.feature[self.current_node],
                                      self.feature[next_node])
        self.total_cost += cost
        done = ops.check_done(self.x)

        self.current_node = next_node
        self.step_count += 1
        self.path.append(self.current_node)
        self.available_node = ops.calculate_available_node(self.node_list,
                                                           self.x)
        self.weight = ops.calculate_weights(self.current_node,
                                            self.feature)
        if len(self.available_node) == 0:
            done = True
        
        return self.x, cost, done

    def __len__(self):
        return self.n_city
    
    @property
    def get_weight(self):
        return self.weight

    @property
    def get_available_node(self):
        return self.available_node
    
    @property
    def get_x(self):
        return self.x

    @property
    def get_path(self):
        return self.path

    @property
    def get_total_cost(self):
        return self.total_cost

    @property
    def get_feature(self):
        return self.feature
