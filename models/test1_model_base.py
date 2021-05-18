import tensorflow as tf
from tensorflow.keras import layers, Model, initializers, activations


class EmbeddingNetwork(Model):
    
    def __init__(self, p):
        super(EmbeddingNetwork, self).__init__()

        self.p = p

        self.theta = layers.Dense(p, input_shape=(None, p))
        self.theta4 = tf.Variable(initializers.GlorotUniform()(shape=(1, p)),
                                  trainable=True,
                                  dtype=tf.float32)
        
        self.relu_for_outputs = layers.ReLU()

    def call(self, x, mu, weight, adj):
        unit1 = x                                           # (N, p) <- (N, p)

        reshape_adj = tf.expand_dims(adj, axis=-1)          # (N, N, 1) <- (N, N)
        ne_mu = tf.math.multiply(mu, reshape_adj)           # (N, N, p) <- (N, p) * (N, N, 1)
        unit2 = tf.reduce_sum(ne_mu, axis=1)                # (N, p) <- (N, N, p)

        ne_weight = tf.math.multiply(weight, adj)           # (N, N) <- (N, N) * (N, N)
        ne_weight = tf.expand_dims(ne_weight, axis=-1)      # (N, N, 1) <- (N, N)
        unit4 = tf.math.multiply(ne_weight, self.theta4)    # (N, N, p) <- (N, N, 1) * (1, p)
        unit4 = tf.reduce_sum(unit4, axis=1)                # (N, p) <- (N, N, p)

        outputs = self.relu_for_outputs(unit1 + unit2 + unit4)  # (N, p) <- (N, p) + (N, p) + (N, p)

        return outputs


class EvaluationNetwork(Model):
    
    def __init__(self, p):
        super(EvaluationNetwork, self).__init__()

        self.p = p

        self.theta5 = layers.Dense(16, input_shape=(None, p * 2),
                                   activation=activations.relu)
        self.dense1 = layers.Dense(32, activation=activations.relu)
        self.dense2 = layers.Dense(64, activation=activations.relu)
        self.output = layers.Dense(1, activation=activations.relu)
        self.concat = layers.Concatenate(axis=1)

    def call(self, sum_mu, mu):
        unit6 = sum_mu
        
        unit7 = mu

        unit5 = self.concat([unit6, unit7])

        x = self.theta5(unit5)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.output(x)

        return outputs
