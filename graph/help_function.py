import numpy as np


class HelpFunction():

    def __init__(self, config):
        self.config = config
        self.weights = None

    def set_weights(self, weights):
        self.weights = weights

    def get_insert_pos(self, path, new_node):
        length = len(path) + 1
        candidate = self._generate_candidate(path, new_node, length)
        dist = self._calculate_cost_on_path(candidate, length)
        idx = np.argmin(dist)

        return idx, dist[idx]

    def _generate_candidate(self, path, new_node, length):
        candidate = np.eye(length, dtype=np.int) * new_node
        p = (np.asarray(path, dtype=int)[np.newaxis, ...] * np.ones((length, 1), dtype=np.int)).flatten()
        indice = self.__indices_without_diag(length)
        candidate[indice] = p

        return candidate
    
    def __indices_without_diag(self, length):
        d0 = (np.arange(length, dtype=np.int)[..., np.newaxis] * np.ones((1, length - 1), dtype=np.int)).flatten()
        d1 = np.arange(1, length + 1, dtype=np.int)[np.newaxis, ...] * np.ones((length - 1, 1), dtype=np.int)
        d1 += np.arange(length - 1)[..., np.newaxis]
        d1 %= length
        return d0, d1.flatten()

    def _calculate_cost_on_path(self, candidate, length):
        idx0 = np.arange(length - 1, dtype=np.int)
        idx1 = idx0 + 1
        d0 = candidate[:, idx0].flatten()
        d1 = candidate[:, idx1].flatten()
        dist = self.weights[d0, d1].reshape(length, length - 1)
        dist = np.sum(dist, axis=-1)
        
        return dist
