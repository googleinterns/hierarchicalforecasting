import tensorflow as tf
import numpy as np
import global_flags

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS

class DFRNN(keras.Model):
    def __init__(self, num_ts):
        super(DFRNN, self).__init__()

        self.num_ts = num_ts
        self.time_steps = flags.cont_len

        self.lstm1 = layers.LSTM(flags.global_lstm_hidden, stateful=True,
                            return_sequences=True, time_major=True)
        self.lstm2 = layers.LSTM(flags.local_lstm_hidden, stateful=True,
                            return_sequences=True, time_major=True)
        self.dense1 = layers.Dense(flags.prin_comp, activation='tanh')
        self.dense2 = layers.Dense(num_ts)
        self.dense3 = layers.Dense(1)

    @tf.function
    def call(self, feats, y_obs):
        '''
        feats: (t+1) x d
        y_obs: (t+1) x n
        '''

        # Computing global effects
        stat_feats = tf.expand_dims(feats[1:], axis=1)  # t x 1 x d
        y_feats = tf.expand_dims(y_obs[:-1], axis=1)  # t x 1 x n
        feats = tf.concat([stat_feats, y_feats], axis=-1)  # t x 1 x D
        
        outputs = self.lstm1(feats)  # t x 1 x h
        outputs = tf.squeeze(outputs, axis=1)  # t x h

        basis = self.dense1(outputs)  # t x k

        global_effect = self.dense2(basis)  # t x n
        global_effect = tf.exp(global_effect)

        # Computing local effects
        y_feats = tf.transpose(y_feats, [0, 2, 1])  # t x n x 1
        stat_feats = tf.repeat(stat_feats, repeats=self.num_ts, axis=1)  # t x n x d
        
        weights = self.dense2.weights
        w = weights[0]  # k x n
        w = tf.transpose(w)  # n x k
        w = tf.expand_dims(w, 0)  # 1 x n x k
        ts_emb = tf.repeat(w, repeats=self.time_steps, axis=0)  # t x n x k
        local_feats = tf.concat([y_feats, stat_feats, ts_emb], axis=-1)  # t x n x D'
        outputs = self.lstm2(local_feats)  # t x n x h'
        sigma = self.dense3(outputs)
        sigma = tf.squeeze(sigma, axis=-1)  # t x n x 1
        sigma = tf.math.log(1 + tf.exp(sigma))

        return global_effect, sigma


@tf.function
def gaussian_nll(mean, sigma, y_obs):
    '''
    mean, sigma: t x n
    y_obs: (t+1) x n
    '''
    y_obs = y_obs[1:] # t x n
    nll = 0.5 * ((y_obs - mean) / sigma)**2 + tf.math.log(sigma)
    nll = tf.reduce_mean(nll)
    return nll
