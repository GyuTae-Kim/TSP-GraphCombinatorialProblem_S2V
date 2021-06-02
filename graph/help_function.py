import numpy as np


class HelpFunction():

    def __init__(self, config):
        self.config = config
        self.weight= None

        self.vec_calculate_dist = np.vectorize(self._calculate_distance,
                                               excluded=['candidate'],
                                               otypes=[object])

    def get_insert_pos(self, path, new_node, weight):
        self.weight = weight
        length = len(path) + 1
        candidate = self._generate_candidate(path, new_node, length)
        dist = self._calculate_cost_on_path(candidate, length)
        idx = np.argmin(dist)

        return idx, dist[idx]

    def _generate_candidate(self, path, new_node, length):
        candidate = np.eye(length, dtype=np.int) * new_node
        index = np.where(candidate == 0)
        p = np.array(path * length, dtype=np.int)
        candidate[index] = p

        return candidate

    def _calculate_cost_on_path(self, candidate, length):
        index = np.arange(length - 1)
        dist = np.array(self.vec_calculate_dist(index=index, candidate=candidate).tolist(), dtype=np.float32)
        dist = np.sum(dist, axis=0)
        
        return dist

    def _calculate_distance(self, index, candidate):
        return self.weight[candidate[:, index], candidate[:, index + 1]].tolist()
