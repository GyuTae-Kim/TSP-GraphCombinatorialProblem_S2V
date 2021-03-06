import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, optimizers

import os

from .test1_model_base import EmbeddingNetwork, EvaluationNetwork
from . import ops


class S2V_DQN(Model):
    
    def __init__(self, config):
        super(S2V_DQN, self).__init__()

        self.config = config

        self.load = config['model_params']['load']
        self.t = config['model_params']['t']
        self.p = config['model_params']['p']
        self.lr = config['model_params']['lr']
        self.save_path = config['train_params']['save_path']
        
        self.G, self.node_list, self.adj, self.feature = None, None, None, None

        print(' [Task] Load S2V')
        self.embedding_net = EmbeddingNetwork(self.p)
        print(' [Done] Successfully Loaded S2V')
        print(' [Task] Load Evaluation(Q)')
        self.evaluate_net = EvaluationNetwork(self.p)
        print(' [Done] Successfully Loadded Evaluation(Q)')
        self.opt = optimizers.Adam(self.lr)

        if self.load:
            print(' [Task] Check Checkpoint')
            self._check_checkpoint()
            print(' [Done] Checking')

    def name(self):
        return 'ModelOnGraph(Test1)'

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
            # assert self.G is None, '  [Err] Import instance first.'
            x = self.G.get_x()
        if adj is None:
            # assert self.G is None, '  [Err] Import instance first.'
            adj = self.adj
        if w is None:
            # assert self.G is None, '  [Err] Import instance first.'
            w = self.G.get_weight()
        if mu is None:
            # assert self.G is None, '  [Err] Import instance first.'
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
            mu = self.embedding_net(x, mu, w, adj)

        return mu

    def evaluate(self, idx, mu):
        sum_mu = tf.reduce_mean(mu, axis=0, keepdims=True)
        brod = tf.convert_to_tensor(np.ones((len(idx), 1)),
                                    dtype=tf.float32)
        sum_mu = sum_mu * brod
        node_mu = ops.specific_value(mu, idx)
        Q = self.evaluate_net(sum_mu, node_mu)

        return Q

    def call(self, idx, x, mu, w, adj):
        emb_mu = self.embedding(x, mu, w, adj)
        Q = self.evaluate(idx, emb_mu)

        return Q

    def update(self, idx, x, mu, weight, adj, opt_Q):
        with tf.GradientTape() as tape:
            Q = self.__call__(idx, x, mu, weight, adj)
            loss = tf.keras.losses.mean_squared_error(opt_Q, Q)
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    def _check_checkpoint(self):
        if not os.path.exists(self.save_path):
            print('  [Done] Couldn\'t find checkpoint')
            return
        latest = tf.train.latest_checkpoint(self.save_path)
        if latest is None:
            print('  [Done] Couldn\'t find checkpoint')
            return
        print('  [Task] Load Weights From {}'.format(latest))
        self.load_weights(latest)
        print('  [Done] Load Checkpoint')
