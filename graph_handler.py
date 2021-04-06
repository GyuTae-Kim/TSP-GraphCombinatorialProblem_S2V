import copy

from utils.memory import Memory as Mem


class GraphHandler(object):
    
    def __init__(self, config, data_gen):
        self.config = config
        self.data_gen = data_gen

        self.batch_size = config['train_params']['batch_size']
        self.memory_size = config['train_params']['memory_size']
        self.train_eps = config['train_params']['max_episode']
        self.test_eps = config['test_params']['max_episode']
        self.total_eps = self.train_eps + self.test_eps
        
        self.data_idx = -1
        self.cur_step = 0
        self.cur_pos = 0
        
        self.mem = Mem(self.memory_size, self.batch_size)
        self.G, self.feature, self.x = None, None, None

    def move_node(self, a):
        next_x, _, done = self.G.move(a)
        w = self.G.get_weight()
        r = self._calculate_cost_tsp(a)
        fail = False

        if self.data_idx < self.train_eps - 1:
            self.mem.append(copy.deepcopy(self.x), a, r, done, w, self.G.get_feature())

        if done:
            if 0. in next_x:
                fail = True
        
        self.x = next_x.copy()

        return done, fail

    def genenrate_train_sample(self):
        return self.mem.sample(self.batch_size)

    def generate_graph_instance(self):
        self.data_idx += 1

        if self.data_idx == self.train_eps - 1:
            self.mem.clear()

        if self.data_idx >= self.total_eps:
            raise IndexError('  [Err] The maximum index of the data generator has been exceeded.')
        
        self.G = self.data_gen[self.data_idx]
        self.feature = self.G.get_feature()
        self.x = self.G.get_x()
        self.cur_step = 0
        self.cur_pos = 0

        return self.G

    def moveable_node(self):
        return self.G.get_available_node()
    
    def _calculate_cost_tsp(self, to_node):
        sum = -self.G.get_total_cost()
        cost = sum - self.G.cost_func(self.feature[0], self.feature[to_node])
        
        return cost
