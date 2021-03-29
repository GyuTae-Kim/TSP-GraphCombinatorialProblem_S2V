import numpy as np

from utils.memory import Memory as Mem


class GraphHandler(object):
    
    def __init__(self, config, data_gen):
        self.config = config
        self.data_gen = data_gen

        self.batch_size = config['train_params']['batch_size']
        self.memory_size = config['train_params']['memory_size']
        self.train_eps = config['train_params']['max_episode']
        self.test_eps = config['test_params']['max_episode']
        self.memory_saving_mode = config['train_params']['memory_saving_mode']
        self.total_eps = self.train_eps + self.test_eps
        
        self.data_idx = -1
        self.cur_step = 0
        self.cur_pos = 0
        
        self.mem = Mem(self.memory_size, self.batch_size)
        self.G, self.feature = None, None

    def move_node(self, a):
        x, _, done = self.G.move(a)
        w = self.G.get_weight()
        r = self._calculate_cost_tsp(a)
        fail = False

        if self.data_idx < self.train_eps - 1:
            self.mem.append(x, a, r, done, w)

        if done:
            if 0. in x:
                fail = True

        return done, fail

    def genenrate_train_sample(self):
        return self.mem.sample()

    def generate_graph_instance(self):
        self.data_idx += 1

        if self.data_idx >= self.total_ep:
            raise IndexError('  [Err] The maximum index of the data generator has been exceeded.')
        
        self.G = self.data_gen[self.data_idx]
        self.feature = self.G.get_feature()
        self.cur_step = 0
        self.cur_pos = 0

        return self.G, len(self.G)

    def moveable_node(self):
        return self.G.get_available_node()

    def _calculate_cost_tsp(self, to_node):
        sum = -self.G.get_total_cost()
        cost = sum - self.G.cost_func(self.feature[0], self.feature[to_node])
        
        return cost
