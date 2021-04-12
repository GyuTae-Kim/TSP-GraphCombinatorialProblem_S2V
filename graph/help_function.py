import numpy as np

import copy

import graph.ops as ops


class HelpFunction():

    def __init__(self, config):
        self.feature = None
        self.config = config

        self.n_key = len(config['data_params']['key'])

    def get_insert_pos(self, path, new_node, feature):
        self.feature = feature
        _path = copy.deepcopy(path)
        _path = self._generate_candidate(_path, new_node)
        cost = [self._calculate_cost_on_path(p) for p in _path]
        idx = np.argmin(cost)

        return idx, cost[idx]

    def _generate_candidate(self, path, new_node):
        length = len(path) + 1
        path *= length
        idx = 0

        for _ in range(length):
            path.insert(idx, new_node)
            idx += length + 1
        
        path = np.array(path, dtype=np.int32)
        path = np.reshape(path, (-1, length))

        return path

    def _calculate_cost_on_path(self, path):
        path_s1, path_s2 = path[:-1], path[1:]
        p1 = np.reshape(self.feature[path_s1, :], (-1, self.n_key))
        p2 = np.reshape(self.feature[path_s2, :], (-1, self.n_key))
        dist = self._calculate_distance_function(p1, p2)
        dist += np.math.sqrt(np.sum((self.feature[0, :] - self.feature[-1, :]) ** 2))

        return dist

    def _calculate_distance_function(self, p1, p2):
        dist = np.sqrt(np.sum((p1 - p2) ** 2, axis=1))
        total_dist = np.sum(dist)

        return total_dist
