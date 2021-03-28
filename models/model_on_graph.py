import tensorflow as tf
from tensorflow.keras import Model, optimizers, losses

from models.model_base import Structure2Vec, Evaluation

import models.ops as ops


class ModelOnGraph(Model):
    
    def __init__(self, config):
        super(ModelOnGraph, self).__init__()

        self.config = config

        self.t = config['model_params']['t']
        self.p = config['model_params']['p']
        self.lr = config['model_params']['lr']
        
        self.G, self.node_list, self.adj, self.feature = None, None, None, None

        print(' [Task] Load S2V')
        self.s2v = Structure2Vec(self.p)
        print(' [Done] Successfully Loaded S2V')
        print(' [Task] Load Evaluation(Q)')
        self.ev = Evaluation(self.p)
        print(' [Done] Successfully Loadded Evaluation(Q)')
        self.opt = optimizers.Adam(self.lr)

    def import_instance(self, G):
        self.G = G
        instance = G.instance_info()
        self.node_list = instance['node_list']
        self.adj = tf.convert_to_tensor(instance['adj'],
                                        dtype=tf.float32)
        self.feature = self.G.get_feature()

    def embedding(self, x, adj=None, w=None):
        assert self.G is None, '  [Err] Import instance first.'

        if adj is None:
            adj = self.adj
        if w is None:
            w = self.G.calculate_weights()

        x = tf.convert_to_tensor(x,
                                 dtype=tf.float32)
        mu = tf.convert_to_tensor(self.feature,
                                  dtype=tf.float32)
        w = tf.convert_to_tensor(w,
                                 dtype=tf.float32)
        for t in range(self.t):
            mu = self.s2v(x, mu, self.adj, w)

        return mu

    def evaluate(self, idx, mu):
        sum_mu = tf.reduce_mean(mu, axis=0, keepdims=True)
        node_mu = ops.specific_value(mu, idx)
        Q = self.ev(sum_mu, node_mu)

        return Q

    def call(self, idx, x, adj=None, weight=None):
        mu = self.embedding(x, adj, weight)
        Q = self.evaluate(idx, mu)

        return Q

    def update(self, idx, x, adj, weight, opt_Q):
        with tf.GradientTape() as tape:
            Q = self.__call__(idx, x, adj, weight)
            loss = losses.mean_squared_error(opt_Q, Q)
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))

        return loss
