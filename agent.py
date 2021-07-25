import numpy as np
import matplotlib.pyplot as plt

import os

import tensorflow as tf
import tensorflow.keras.backend as K

import ops
from graph_handler import GraphHandler


class Agent(object):

    def __init__(self, config, graph_handler, s2v_dqn):
        self.config = config
        self.graph_handler = graph_handler
        self.s2v_dqn = s2v_dqn
        self.test_data_gen = None

        self.batch_size = config['train_params']['batch_size']
        self.train_eps = config['train_params']['max_episode']
        self.train_epoch = config['train_params']['train_epoch']
        self.discount = config['train_params']['discount']
        self.save_path = config['train_params']['save_path']
        self.save_freq = config['train_params']['save_freq']
        self.n_step = config['train_params']['n_step']
        self.test_while_training = config['train_params']['test_while_training']
        self.test_freq = config['train_params']['test_freq']
        self.test_eps = config['test_params']['max_episode']
        self.save_test_log = config['test_params']['save_test_log']
        if config['test_params']['save_test_log']:
            self.test_result_path = config['test_params']['test_result_path']
        else:
            self.test_result_path = None
        self.discount = K.variable(self.discount)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.checkpoint_format = os.path.join(self.save_path, "cp-{epoch:04d}.ckpt")

        self.loss = []

    def running(self):
        self.run_train()
        self.run_test()
        
    def run_train(self):
        print('[Task] Start Train')
        for ep in range(self.train_eps):
            G = self.graph_handler.generate_graph_instance()
            self.s2v_dqn.import_instance(G)
            e = 1. / ((ep / 10) + 1)
            done = False
            step = 0
            ep_loss = []

            while not done:
                moveable_node = self.graph_handler.moveable_node()
                if np.random.rand(1) < e:
                    a = np.random.choice(moveable_node)
                else:
                    Q = self.get_Q_value(moveable_node)
                    a = moveable_node[np.argmax(Q)]

                done = self.graph_handler.move_node(a)
                step += 1

                if step >= self.n_step:
                    if self.graph_handler.is_available_train():
                        loss = self.update_model()
                        ep_loss.append(loss)

            if len(ep_loss) > 0:
                ep_avg_loss = np.mean(ep_loss)
                self.loss.append(ep_avg_loss)
            else:
                ep_avg_loss = None
            print(' [Train] Ep: {}/{} Step: {} Cost: {} Loss: {}'.format(ep,
                                                                         self.train_eps,
                                                                         step,
                                                                         self.graph_handler.bef_cost,
                                                                         ep_avg_loss))

            if ep % self.save_freq == 0 and ep != 0:
                self.save_model_weights(ep)
                print(' [Done] Saved model')
            
            if ep % self.test_freq == 0 and ep != 0:
                self.run_test_while_training(ep)
                

        self.save_model_weights(self.train_eps)
        self.save_loss_plt()

    def run_test(self):
        print('[Task] Start Test')
        cost = []

        if self.save_test_log:
            if not os.path.exists(self.test_result_path):
                f = open(self.test_result_path, 'w')
                f.close()

        for e in range(self.test_eps):
            G = self.graph_handler.generate_graph_instance()
            self.s2v_dqn.import_instance(G)
            done = False
            n_visit = 1

            while not done:
                moveable_node = self.graph_handler.moveable_node()
                Q = self.get_Q_value(moveable_node)
                a = moveable_node[np.argmax(Q)]
                done = self.graph_handler.move_node(a)
                n_visit += 1

            total_cost = self.graph_handler.bef_cost
            cost.append(total_cost)
            
            print(' [Test] Ep: {}/{}, cost: {}'.format(e, self.test_eps, total_cost))
            
            if self.save_test_log and self.graph_handler.saving:
                with open(self.test_result_path, 'a') as f:
                    data = '{} {}\n'.format(G.n_city, total_cost)
                    f.write(data)
            elif self.save_test_log and self.test_while_training:
                with open(self.graph_handler.get_result_path):
                    data = '{} {}\n'.format(G.n_city, total_cost)
                    f.write(data)
    
    def run_test_while_training(self, ep):
        test_graph_handler = GraphHandler(self.config, self.test_data_gen, None)
        test_graph_handler.set_saving_mode(False)
        test_graph_handler.set_result_path(os.path.join('results', '{}_test.txt'.format(ep)))
        temp = self.graph_handler
        self.graph_handler = test_graph_handler
        self.run_test()
        self.graph_handler = temp

    def get_Q_value(self, moveable_node):
        mu = self.s2v_dqn.embedding()
        Q = self.s2v_dqn.evaluate(moveable_node, mu).numpy()

        return Q

    def update_model(self):
        loss = []

        for e in range(self.train_epoch):
            batch_G_idx, batch_S, batch_v, batch_R, batch_W = self.graph_handler.genenrate_train_sample()
            
            for i, S, v, R, W in zip(batch_G_idx, batch_S, batch_v, batch_R, batch_W):
                S, future_S = S[0], S[1]
                G = self.graph_handler.get_instance(i)
                A = G.get_adjacency_matrix()
                R = tf.convert_to_tensor(R, dtype=tf.float32)

                Q = tf.convert_to_tensor([[0.]], dtype=tf.float32)
                Q += R + self.discount * K.max(self.s2v_dqn(ops.calculate_available_node(future_S),
                                                            future_S,
                                                            W,
                                                            A))
                loss.append(self.s2v_dqn.update([v], S, W, A, Q))
        
        return np.mean(loss)
    
    def set_test_data_gen(self, data_gen):
        self.test_data_gen = data_gen

    def save_model_weights(self, ep):
        self.s2v_dqn.save_weights(self.checkpoint_format.format(epoch=ep))
    
    def save_loss_plt(self):
        plt.plot(self.loss)
        plt.ylabel('Loss')
        plt.savefig('results/loss.png')

        self.loss.clear()
