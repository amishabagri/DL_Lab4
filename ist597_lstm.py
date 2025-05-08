import os
import numpy as np
import tensorflow as tf
import random
import json
import matplotlib.pyplot as plt

# Set consistent seed
def set_seed(seed=0):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class BasicLSTM(tf.keras.Model):
    def __init__(self, units, return_sequence=False, return_states=False, **kwargs):
        super(BasicLSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequence = return_sequence
        self.return_states = return_states

        def bias_initializer(shape, dtype=None):
            return tf.concat([
                tf.zeros((self.units,), dtype=dtype),  # input gate
                tf.ones((self.units,), dtype=dtype),   # forget gate
                tf.zeros((self.units * 2,), dtype=dtype),  # cell/output gates
            ], axis=0)

        self.kernel = tf.keras.layers.Dense(4 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(
            4 * units, kernel_initializer='glorot_uniform', bias_initializer=bias_initializer
        )

    def call(self, inputs, training=None, mask=None, initial_states=None):
        if initial_states is None:
            h_state = tf.zeros((tf.shape(inputs)[0], self.units))
            c_state = tf.zeros((tf.shape(inputs)[0], self.units))
        else:
            h_state, c_state = initial_states

        h_list = []
        c_list = []

        for t in range(tf.shape(inputs)[1]):
            ip = inputs[:, t, :]
            z = self.kernel(ip) + self.recurrent_kernel(h_state)

            i = tf.keras.activations.sigmoid(z[:, :self.units])
            f = tf.keras.activations.sigmoid(z[:, self.units:2*self.units])
            c_hat = tf.nn.tanh(z[:, 2*self.units:3*self.units])
            o = tf.keras.activations.sigmoid(z[:, 3*self.units:])

            c_state = f * c_state + i * c_hat
            h_state = o * tf.nn.tanh(c_state)

            h_list.append(h_state)
            c_list.append(c_state)

        hidden_outputs = tf.stack(h_list, axis=1)

        if self.return_states and self.return_sequence:
            return hidden_outputs, [hidden_outputs, tf.stack(c_list, axis=1)]
        elif self.return_states:
            return hidden_outputs[:, -1, :], [h_state, c_state]
        elif self.return_sequence:
            return hidden_outputs
        else:
            return hidden_outputs[:, -1, :]
