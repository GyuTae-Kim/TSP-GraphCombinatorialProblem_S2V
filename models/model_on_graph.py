import tensorflow as tf
from tensorflow.keras import Model, optimizers

import os

from .model_base import Structure2Vec, Evaluation
from . import ops


class ModelOnGraph(Model):
    
    def __init__(self, config):
        super(ModelOnGraph, self).__init__()

        self.config = config

        self.load = config['model_params']['load']
        self.t = config['model_params']['t']
        self.p = config['model_params']['p']
        self.init_lr = config['model_params']['init_lr']
        self.save_path = config['train_params']['save_path']
        self.decay_steps = config['model_params']['decay_steps']
        self.decay_rate = config['model_params']['decay_rate']
        self.grad_clip = config['model_params']['grad_clip']
        
        self.G, self.node_list, self.adj, self.feature = None, None, None, None

        print(' [Task] Load S2V')
        self.s2v = Structure2Vec(self.p)
        print(' [Done] Successfully Loaded S2V')
        print(' [Task] Load Evaluation(Q)')
        self.ev = Evaluation(self.p)
        print(' [Done] Successfully Loadded Evaluation(Q)')
        self.global_step = tf.Variable(0, trainable=False)
        lr_decay = tf.compat.v1.train.exponential_decay(self.init_lr,
                                                        self.global_step,
                                                        self.decay_steps,
                                                        self.decay_rate,
                                                        staircase=True)
        self.opt = tf.compat.v1.train.AdamOptimizer(lr_decay)

        if self.load:
            print(' [Task] Check Checkpoint')
            self._check_checkpoint()
            print(' [Done] Checking')
    
    def name(self):
        return 'ModelOnGraph(S2V)'

    def import_instance(self, G):
        if G is None:
            ValueError('  [Err] Graph instance couldn\'t be None Value.')

        self.G = G
        instance = G.instance_info()
        self.node_list = instance['node_list']
        self.adj = instance['adj']
        self.feature = self.G.get_feature()

    def embedding(self, x=None, mu=None, w=None, adj=None):
        if x is None:
            x = self.G.get_x()
        if adj is None:
            adj = self.adj
        if w is None:
            w = self.G.get_weight()
        if mu is None:
            mu = self.feature

        x = tf.convert_to_tensor(x,
                                 dtype=tf.float32)
        adj = tf.convert_to_tensor(adj,
                                   dtype=tf.float32)
        w = tf.convert_to_tensor(w,
                                 dtype=tf.float32)
        mu = tf.convert_to_tensor(mu,
                                  dtype=tf.float32)

        for t in range(self.t):
            mu = self.s2v(x, mu, w, adj)

        return mu

    def evaluate(self, idx, mu):
        sum_mu = tf.reduce_sum(mu, axis=0, keepdims=True)
        brod = tf.ones((len(idx), 1), dtype=tf.float32)
        sum_mu = sum_mu * brod
        node_mu = ops.specific_value(mu, idx)
        Q = self.ev(sum_mu, node_mu)

        return Q

    def call(self, idx, x, mu, w, adj):
        emb_mu = self.embedding(x, mu, w, adj)
        Q = self.evaluate(idx, emb_mu)

        return Q

    def update(self, idx, x, mu, weight, adj, opt_Q):
        with tf.GradientTape() as tape:
            Q = self.__call__(idx, x, mu, weight, adj)
            loss = tf.keras.losses.mean_squared_error(opt_Q, Q)
        tvars = self.trainable_variables
        grads = tape.gradient(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, 5)
        self.opt.apply_gradients(zip(grads, tvars), self.global_step)
        self.global_step.assign_add(1)

        return loss

    def _check_checkpoint(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            print('  [Done] Couldn\'t find checkpoint')
            return
        latest = tf.train.latest_checkpoint(self.save_path)
        if latest is None:
            print('  [Done] Couldn\'t find checkpoint')
            return
        print('  [Task] Load Weights From {}'.format(latest))
        self.load_weights(latest)
        print('  [Done] Load Checkpoint')
