import numpy as np

import graph.ops as ops


class Instance(object):

    def __init__(self, node_list, feature_x, feature_y, weight=None, connected_all=True):
        if not (node_list.shape == feature_x.shape == feature_y.shape):
            raise ValueError('   [Err] Parameters\' shape are not match.'
                             'city_list: {}, feature_x: {}, feature_y'.format(node_list.shape,
                                                                              feature_x.shape,
                                                                              feature_y.shape))

        self.node_list = node_list
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.weight = weight
        self.connected_all = connected_all

        self.node_count = len(node_list)
        self.A = ops.gen_adjacency_matrix(self.node_count)
        self.x = ops.gen_init_x(self.node_count)

        if weight is None:
            self.w = self.A

        self.current_node = node_list[0]
        self.available_node = ops.calculate_available_node(node_list,
                                                           self.x,
                                                           connected_all=connected_all,
                                                           A=self.A[self.current_node])

        self.total_cost = 0.
        self.step_count = 0
        self.path = [self.current_node]

    def move(self, next_node):
        if next_node not in self.available_node:
            raise ValueError('   [Err] Unable to move to that node.'
                             'now: {} dest: {}'.format(self.current_node,
                                                       next_node))
        
        self.x = ops.calculate_x(self.x, next_node)
        cost = self.w[self.current_node][next_node]
        self.total_cost += cost
        done = ops.check_done(self.x)

        self.current_node = next_node
        self.step_count += 1
        self.path.append(self.current_node)
        self.available_node = ops.calculate_available_node(self.node_list,
                                                           self.x,
                                                           connected_all=self.connected_all,
                                                           A=self.A[self.current_node])
        if len(self.available_node) == 0:
            done = True
        
        return self.x, cost, done

    @property
    def get_path(self):
        return self.path

    @property
    def get_total_cost(self):
        return self.total_cost
    
    @property
    def get_adjecency_matrix(self):
        return self.A
    
    @property
    def get_weight(self):
        return self.w
