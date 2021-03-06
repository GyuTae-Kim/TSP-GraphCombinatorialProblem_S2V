import tensorflow as tf
from tensorflow.keras import Model

import os

from .model_base import Structure2Vec, Evaluation
from . import ops


class S2V_DQN(Model):
    
    def __init__(self, config):
        super(S2V_DQN, self).__init__()

        self.config = config

        self.load = config['model_params']['load']
        self.t = config['model_params']['t']
        self.p = config['model_params']['p']
        self.init_lr = config['model_params']['init_lr']
        self.save_path = config['train_params']['save_path']
        self.decay_steps = config['model_params']['decay_steps']
        self.decay_rate = config['model_params']['decay_rate']
        self.grad_clip = config['model_params']['grad_clip']
        self.node_feat_size = config['data_params']['node_feat_size']
        self.edge_feat_size = config['data_params']['edge_feat_size']
        
        self.G, self.node_list, self.adj = None, None, None

        print(' [Task] Load S2V')
        self.s2v = Structure2Vec(self.p, self.node_feat_size, self.edge_feat_size)
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
        return 'S2V-DQN'

    def import_instance(self, G):
        if G is None:
            ValueError('  [Err] Graph instance couldn\'t be None Value.')

        self.G = G
        instance = G.instance_info()
        self.node_list = instance['node_list']
        self.adj = instance['adj']

    def embedding(self, node_feat=None, edge_feat=None, adj=None):
        if node_feat is None:
            node_feat = self.G.get_nodefeat()
        if adj is None:
            adj = self.adj
        if edge_feat is None:
            edge_feat = self.G.get_edgefeat()

        node_feat = tf.convert_to_tensor(node_feat, dtype=tf.float32)
        adj = tf.convert_to_tensor(adj, dtype=tf.float32)
        edge_feat = tf.convert_to_tensor(edge_feat, dtype=tf.float32)
        mu = tf.zeros((node_feat.shape[0], self.p), dtype=tf.float32)

        for _ in range(self.t):
            mu = self.s2v(node_feat, mu, edge_feat, adj)

        return mu

    def evaluate(self, idx, mu):
        sum_mu = tf.reduce_sum(mu, axis=0, keepdims=True)
        brod = tf.ones((len(idx), 1), dtype=tf.float32)
        sum_mu = sum_mu * brod
        node_mu = ops.specific_value(mu, idx)
        Q = self.ev(sum_mu, node_mu)

        return Q

    def call(self, idx, node_feat, edge_feat, adj):
        mu = self.embedding(node_feat, edge_feat, adj)
        Q = self.evaluate(idx, mu)

        return Q

    def update(self, idx, node_feat, edge_feat, adj, opt_Q):
        with tf.GradientTape() as tape:
            Q = self.__call__(idx, node_feat, edge_feat, adj)
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
