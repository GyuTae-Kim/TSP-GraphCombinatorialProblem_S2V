import numpy as np

import os

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
        self.save_freq = config['train_params']['save_eq']
        self.test_eps = config['test_params']['max_episode']

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.checkpoint_format = os.path.join(self.save_path, 'model-{epoch:04d}.ckpt')

        self.avg_loss = []
        
    def run_train(self):
        for e in range(self.train_eps):
            G, node_count = self.graph_handler.generate_graph_instance()
            self.model_on_graph.import_instance(G)
            e = 1. / ((e / 10) + 1)
            done = False
            n_visit = 1

            while not done:
                moveable_node = self.graph_handler.moveable_node()
                if np.random.rand(1) < e:
                    a = np.random.choice(moveable_node)
                else:
                    Q = self.get_Q_value(moveable_node)
                    a = moveable_node[np.argmax(Q)]

                self.graph_handler.move_node(a)
                n_visit += 1

            if e % self.update_freq == 0 and e != 0:
                loss = self.update_model()
                print(' [Done] Ep: {}/{}, Update Model. Loss: {}'.format(e,
                                                                         self.train_eps,
                                                                         loss))
            if e % self.save_freq == 0 and e != 0:
                self.save_model_weight()
                print(' [Done] Save model')

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
                Q = self.model_on_graph([a], x, adj, w, f).numpy()

                if done:
                    Q[0, 0] = r
                else:
                    next_x = x.copy(); next_x[a] = 1.
                    Q[0, 0] = r + self.discount * np.max(self.model_on_graph(ops.calculate_available_node(next_x),
                                                                             next_x,
                                                                             adj,
                                                                             w,
                                                                             f))
                loss.append(self.model_on_graph.update([a], x, adj, w, f, Q))
        
        return np.mean(loss)

    def save_model_weight(self):
        self.model_on_graph.save_weights(self.checkpoint_format)
