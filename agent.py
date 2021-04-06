import numpy as np

import os
import copy

import ops


class Agent(object):

    def __init__(self, config, graph_handler, model_on_graph):
        self.config = config
        self.graph_handler = graph_handler
        self.model_on_graph = model_on_graph

        self.batch_size = config['train_params']['batch_size']
        self.train_eps = config['train_params']['max_episode']
        self.update_freq = config['train_params']['update_freq']
        self.train_epoch = config['train_params']['train_epoch']
        self.discount = config['train_params']['discount']
        self.save_path = config['train_params']['save_path']
        self.save_freq = config['train_params']['save_freq']
        self.test_eps = config['test_params']['max_episode']

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.checkpoint_format = os.path.join(self.save_path, 'latest_weights.ckpt')

        self.avg_loss = []

    def running(self):
        self.run_train()
        self.run_test()
        
    def run_train(self):
        print('[Task] Start Train')
        for ep in range(self.train_eps):
            G = self.graph_handler.generate_graph_instance()
            self.model_on_graph.import_instance(G)
            e = 1. / ((ep / 10) + 1)
            done = False
            n_visit = 1

            while not done:
                moveable_node = self.graph_handler.moveable_node()
                if np.random.rand(1) < e:
                    a = np.random.choice(moveable_node)
                else:
                    Q = self.get_Q_value(moveable_node)
                    a = moveable_node[np.argmax(Q)]

                done, fail = self.graph_handler.move_node(a)
                n_visit += 1
            print(' [Done] Ep: {}/{}'.format(ep, self.train_eps))

            if ep % self.update_freq == 0 and ep != 0:
                loss = self.update_model()
                print(' [Done] Ep: {}/{}, Update Model. Loss: {} / fail: {}'.format(ep,
                                                                             self.train_eps,
                                                                             loss,
                                                                             fail))
            if ep % self.save_freq == 0 and ep != 0:
                self.save_model_weights()
                print(' [Done] Save model')
        self.save_model_weights()

    def run_test(self):
        for e in range(self.test_eps):
            G = self.graph_handler.generate_graph_instance()
            self.model_on_graph.import_instance(G)
            done = False
            n_visit = 1

            while not done:
                moveable_node = self.graph_handler.moveable_node()
                Q = self.get_Q_value(moveable_node)
                a = moveable_node[np.argmax(Q)]
                self.graph_handler.move_node(a)
                n_visit += 1

            print(' [Test] Ep: {}/{}, cost: {}'.format(e, self.test_eps, G.total_cost))

    def get_Q_value(self, moveable_node):
        mu = self.model_on_graph.embedding()
        Q = self.model_on_graph.evaluate(moveable_node, mu).numpy()

        return Q

    def update_model(self):
        loss = []

        for e in range(self.train_epoch):
            b_x, b_a, b_r, b_done, b_w, b_f = self.graph_handler.genenrate_train_sample()
            
            for x, a, r, done, w, f in zip(b_x, b_a, b_r, b_done, b_w, b_f):
                adj = ops.gen_adjacency_matrix(len(x))
                Q = self.model_on_graph([a], x, f, w, adj).numpy()

                if done:
                    Q[0, 0] = r
                else:
                    next_x = copy.deepcopy(x)
                    next_x[a, 0] = 1.
                    Q[0, 0] = r + self.discount * np.max(self.model_on_graph(ops.calculate_available_node(next_x),
                                                                             next_x,
                                                                             f,
                                                                             w,
                                                                             adj))
                loss.append(self.model_on_graph.update([a], x, f, w, adj, Q))
        
        return np.mean(loss)

    def save_model_weights(self):
        self.model_on_graph.save_weights(self.checkpoint_format)
