import numpy as np


class Memory(object):

    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.x = []
        self.a = []
        self.r = []
        self.done = []
        self.w = []
        self.f = []
    
    def append(self, x, a, r, done, w, f):
        self.x.append(x)
        self.a.append(a)
        self.r.append(r)
        self.done.append(done)
        self.w.append(w)
        self.f.append(f)

        if len(self.x) > self.memory_size:
            self.x.pop(0)
            self.a.pop(0)
            self.r.pop(0)
            self.done.pop(0)
            self.w.pop(0)
            self.f.pop(0)

    def clear(self):
        self.x.clear()
        self.a.clear()
        self.r.clear()
        self.done.clear()
        self.w.clear()
        self.f.clear()

    def sample(self, batch_size):
        assert batch_size < len(self.x), '   [err] Data is less than batch size. data length: {}'.format(len(self.x))

        idx = np.random.choice(np.arange(len(self.x)),
                               batch_size,
                               replace=False)
        
        batch_x = np.array(self.x)[idx]
        batch_a = np.array(self.a, dtype=np.float32)[idx]
        batch_r = np.array(self.r, dtype=np.float32)[idx]
        batch_done = np.array(self.done, dtype=np.float32)[idx]
        batch_w = np.array(self.w, dtype=np.float32)[idx]
        batch_f = np.array(self.f, dtype=np.float32)[idx]

        return (batch_x, batch_a, batch_r, batch_done, batch_w, batch_f)
