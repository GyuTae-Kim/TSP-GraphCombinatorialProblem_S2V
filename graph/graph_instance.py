import numpy as np

import graph.ops as ops


class Instance(object):

    def __init__(self, config, node_list, feature):
        if not (node_list.shape[0] == feature.shape[0]):
            raise ValueError('   [Err] Parameters\' length are not match.'
                             'city_list: {}, feature: {}'.format(node_list.shape,
                                                                 feature.shape))

        self.config = config
        self.node_list = node_list
        self.feature = feature

        self.node_count = len(node_list)
        self.A = ops.gen_adjacency_matrix(self.node_count)
        self.x = ops.gen_init_x(self.node_count)

        self.current_node = node_list[0]
        self.available_node = ops.calculate_available_node(node_list,
                                                           self.x)

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
        if len(self.available_node) == 0:
            done = True
        
        return self.x, cost, done

    def calculate_weights(self):
        return ops.calculate_weights(self.current_node, self.feature)
    
    @property
    def get_path(self):
        return self.path

    @property
    def get_total_cost(self):
        return self.total_cost

    @property
    def get_feature(self):
        return self.feature
