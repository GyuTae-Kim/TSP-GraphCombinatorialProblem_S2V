import numpy as np


class Memory(object):

    def __init__(self, config, data_gen):
        self.config = config
        self.data_gen = data_gen

        self.memory_size = config['train_params']['memory_size']
        self.batch_size = config['train_params']['batch_size']

        self.g_idx = []
        self.S = []
        self.v = []
        self.R = []
        self.W = []

        self.cur_idx = 0
    
    def set_index(self, idx):
        self.cur_idx = idx
    
    def append(self, S, v, R, W):
        self.g_idx.append(self.cur_idx)
        self.S.append(S)
        self.v.append(v)
        self.R.append(R)
        self.W.append(W)

        if len(self.g_idx) > self.memory_size:
            self.g_idx.pop(0)
            self.S.pop(0)
            self.v.pop(0)
            self.R.pop(0)
            self.W.pop(0)

    def clear(self):
        self.g_idx.clear()
        self.S.clear()
        self.v.clear()
        self.R.clear()
        self.W.clear()
    
    def sample(self):
        data_len = len(self.g_idx)
        assert self.batch_size - 1 < data_len, '   [err] Data is less than batch size. data length: {} batch_size: {}'.format(data_len, self.batch_size)

        idx = np.random.choice(np.arange(data_len),
                               self.batch_size,
                               replace=False)
        
        batch_G_idx = np.array(self.g_idx, dtype=np.int32)[idx]
        batch_S = np.array(self.S, dtype=np.object)[idx]
        batch_v = np.array(self.v, dtype=np.int32)[idx]
        batch_R = np.array(self.R, dtype=np.float32)[idx]
        batch_W = np.array(self.W, dtype=np.object)[idx]

        return (batch_G_idx, batch_S, batch_v, batch_R, batch_W)
    
    def __len__(self):
        return len(self.g_idx)
