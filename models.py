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

        self.lstm1 = layers.LSTM(flags.global_lstm_hidden,
                            return_sequences=True, time_major=True)
        self.lstm2 = layers.LSTM(flags.local_lstm_hidden,
                            return_sequences=True, time_major=True)
        self.dense1 = layers.Dense(flags.prin_comp, activation='tanh')
        self.dense2 = layers.Dense(num_ts)
        self.dense3 = layers.Dense(1)

    def call(self, feats, y_prev):
        '''
        feats: t x d
        y_prev: t x n
        '''

        # Computing global effects
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        y_feats = tf.expand_dims(y_prev, axis=1)  # t x 1 x n
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
    def train_step(self, feats, y_obs, optimizer):
        with tf.GradientTape() as tape:
            mean, sig = self(feats[1:], y_obs[:-1])
            loss = gaussian_nll(mean, sig, y_obs[1:])
        
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        abs_err = tf.reduce_mean(tf.abs(mean - y_obs[1:]))
        return loss, abs_err
    
    def forecast(self, feats, y_prev):
        '''
        feats: (t+p-1) x n
        y_prev: t x n
        '''
        cont_len = flags.cont_len
        pred_hor = flags.pred_hor
        
        for i in range(pred_hor):
            mean, sig = self.call(feats[i:i+cont_len], y_prev[i:i+cont_len])
            mean = mean[-1]
            sig = sig[-1]
            sample = mean + sig * tf.random.normal(mean.shape)
            sample = np.clip(sample.numpy(), a_min=0, a_max=None).reshape((1, -1))
            y_prev = np.concatenate([y_prev, sample], axis=0)
        
        return y_prev[-pred_hor:]

    def eval(self, dataset):
        cont_len = flags.cont_len
        pred_hor = flags.pred_hor

        diffs = []
        for feats, y_obs in dataset:
            feats = feats.numpy()
            y_obs = y_obs.numpy()
            cont_len = flags.cont_len
            pred_hor = flags.pred_hor
            
            assert(y_obs.shape[0] == cont_len + pred_hor)
            assert(feats.shape[0] == cont_len + pred_hor)

            samples = []
            for i in range(100):
                y_pred = self.forecast(feats[1:], y_obs[:cont_len])
                samples.append(y_pred)
            medians = np.median(samples, axis=0)
            diffs.append(np.abs(medians - y_obs[-pred_hor:]))
        
        return {'q50': np.mean(diffs)}


@tf.function
def gaussian_nll(mean, sigma, y_obs):
    '''
    mean, sigma: t x n
    y_obs: t x n
    '''
    nll = 0.5 * ((y_obs - mean) / sigma)**2 + tf.math.log(sigma)
    nll = tf.reduce_mean(nll)
    return nll
