import tensorflow as tf
from tensorflow.keras import layers, Model, initializers


class Structure2Vec(Model):
    
    def __init__(self, p, node_feat_size, edge_feat_size):
        super(Structure2Vec, self).__init__()

        self.p = p
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size

        self.theta1 = layers.Dense(p, input_shape=(None, self.node_feat_size))
        self.theta2 = layers.Dense(p, input_shape=(None, p))
        self.theta3 = layers.Dense(p, input_shape=(None, p))
        self.theta4 = tf.Variable(initializers.GlorotUniform()(shape=(self.edge_feat_size, p)),
                                  trainable=True,
                                  dtype=tf.float32)
        
        self.relu_unit4 = layers.ReLU()
        self.relu_outputs = layers.ReLU()

    def call(self, node_feat, mu, edge_feat, adj):
        unit1 = self.theta1(node_feat)

        reshape_adj = tf.expand_dims(adj, axis=-1)
        ne_mu = tf.math.multiply(mu, reshape_adj)
        unit2 = tf.reduce_sum(ne_mu, axis=1)
        unit2 = self.theta2(unit2)

        ne_edge_feat = tf.math.multiply(edge_feat, reshape_adj)
        unit4 = tf.matmul(ne_edge_feat, self.theta4)
        unit4 = self.relu_unit4(unit4)
        unit4 = tf.reduce_sum(unit4, axis=1)

        unit3 = self.theta3(unit4)

        outputs = self.relu_outputs(unit1 + unit2 + unit3)

        return outputs


class Evaluation(Model):
    
    def __init__(self, p):
        super(Evaluation, self).__init__()

        self.p = p

        self.theta5 = layers.Dense(1, input_shape=(None, p * 2))
        self.theta6 = layers.Dense(p, input_shape=(None, p))
        self.theta7 = layers.Dense(p, input_shape=(None, p))
        self.concat = layers.Concatenate(axis=1)
        self.relu = layers.ReLU()

    def call(self, sum_mu, mu):
        unit6 = self.theta6(sum_mu)
        
        unit7 = self.theta7(mu)

        unit5 = self.concat([unit6, unit7])
        unit5 = self.relu(unit5)

        outputs = self.theta5(unit5)

        return outputs
