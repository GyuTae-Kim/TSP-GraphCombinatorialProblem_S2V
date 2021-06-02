import numpy as np

from utils.memory import Memory as Mem


class GraphHandler(object):
    
    def __init__(self, config, data_gen, mem):
        self.config = config
        self.data_gen = data_gen
        self.mem = mem

        self.memory_size = config['train_params']['memory_size']
        self.train_eps = config['train_params']['max_episode']
        self.batch_size = config['train_params']['batch_size']
        self.n_step = config['train_params']['n_step']
        self.test_eps = config['test_params']['max_episode']
        self.use_help_func = config['data_params']['use_help_func']
        self.total_eps = self.train_eps + self.test_eps
        
        self.data_idx = -1
        self.cur_step = 0
        self.bef_cost = 0.

        self.G, self.feature, self.weight = None, None, None

        self.saving = True
        self.result_path = None

        self.r = []
        self.S = []
        self.v = []

        if self.use_help_func:
            from graph.help_function import HelpFunction as H
            self.H = H(config)

    def move_node(self, a):
        x = self.G.get_x()
        if self.use_help_func:
            idx, cost = self.H.get_insert_pos(self.G.path, a, self.weight)
            done = self.G.move(a, idx)
            r = self.bef_cost - cost
            self.bef_cost = cost
        else:
            done = self.G.move(a)
            r = self._calculate_reward_tsp(a)
        self.cur_step += 1

        if self.data_idx < self.train_eps and self.saving:
            self.r.append(r); self.S.append(x); self.v.append(a)
            if self.cur_step >= self.n_step:
                bef_idx = self.cur_step - self.n_step
                self.mem.append(S=(self.S[bef_idx], x),
                                v=self.v[bef_idx],
                                R=np.sum(self.r[bef_idx:self.cur_step], dtype=np.float32))

        return done

    def genenrate_train_sample(self):
        return self.mem.sample()

    def generate_graph_instance(self):
        self.data_idx += 1

        if self.data_idx == self.train_eps - 1:
            self.mem.clear()

        if self.data_idx >= self.total_eps:
            raise IndexError('  [Err] The maximum index of the data generator has been exceeded.')
        
        self.G = self.data_gen[self.data_idx]
        self.feature = self.G.get_feature()
        self.weight = self.G.get_weight()
        self.cur_step = 0
        self.bef_cost = 0.

        if self.data_idx <= self.train_eps:
            self.r, self.S, self.v = [], [], []
            self.mem.set_index(self.data_idx)

        return self.G

    def moveable_node(self):
        return self.G.get_available_node()
    
    def get_instance(self, idx):
        return self.data_gen[idx]
    
    def set_result_path(self, path):
        self.result_path = path
    
    def get_result_path(self):
        return self.result_path
    
    def _calculate_reward_tsp(self, to_node):
        cost = self.G.get_total_cost() + self.G.cost_func(self.feature[0], self.feature[to_node])
        r = self.bef_cost - cost
        self.bef_cost = cost
        
        return r

    def is_available_train(self):
        return False if len(self.mem) < self.batch_size else True

    def set_saving_mode(self, is_saving):
        self.saving = is_saving
