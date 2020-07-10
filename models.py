import tensorflow as tf
import numpy as np
import global_flags
import sys

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
        self.feat_emb = layers.Dense(flags.feat_emb_dim, activation='tanh')
        self.dense2 = layers.Dense(num_ts)
        self.dense3 = layers.Dense(1)

    @tf.function
    def call(self, feats, y_prev):
        '''
        feats: t x d
        y_prev: t x n
        '''

        # Computing global effects
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        y_feats = tf.expand_dims(y_prev, axis=1)  # t x 1 x n
        y_emb = self.feat_emb(y_feats)  # t x 1 x e
        feats = tf.concat([stat_feats, y_emb], axis=-1)  # t x 1 x D
        
        outputs = self.lstm1(feats)  # t x 1 x h
        outputs = tf.squeeze(outputs, axis=1)  # t x h

        basis = tf.tanh(outputs)

        global_effect = self.dense2(basis)  # t x n

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
        local_effect = self.dense3(outputs)  # t x n x 1
        local_effect = tf.squeeze(local_effect, axis=-1)
        
        final_output = tf.math.softplus(global_effect + local_effect)

        return final_output
    
    @tf.function
    def train_step(self, feats, y_obs, optimizer):
        with tf.GradientTape() as tape:
            pred = self(feats[1:], y_obs[:-1])
            loss = tf.reduce_mean(tf.math.square(pred - y_obs[1:]))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss
    
    def forecast(self, feats, y_prev):
        '''
        feats: (t+p-1) x n
        y_prev: t x n
        '''
        cont_len = flags.cont_len
        pred_hor = flags.pred_hor
        pred_path = y_prev
        
        for i in range(pred_hor):
            pred = self.call(feats[i:i+cont_len],pred_path[i:i+cont_len])
            pred = pred[-1]
            pred = np.clip(pred.numpy(), a_min=0, a_max=None).reshape((1, -1))
            pred_path = np.concatenate([pred_path, pred], axis=0)
        
        return pred_path[-pred_hor:]

    def eval(self, dataset):
        cont_len = flags.cont_len
        pred_hor = flags.pred_hor

        for feats, y_obs in dataset:
            feats = feats.numpy()
            y_obs = y_obs.numpy()
            
            assert(y_obs.shape[0] == cont_len + pred_hor)
            assert(feats.shape[0] == cont_len + pred_hor)

            y_pred = self.forecast(feats[1:], y_obs[:cont_len])
            diff = np.square(y_pred - y_obs[-pred_hor:])
            rmse = np.sqrt(np.mean(diff, axis=0))
            rmse = np.mean(rmse)

        np.save('notebooks/evals.npy', y_pred)
        return {
            'test/rmse': rmse
        }


@tf.function
def gaussian_nll(mean, sigma, y_obs):
    '''
    mean, sigma: t x n
    y_obs: t x n
    '''
    nll = 0.5 * ((y_obs - mean) / sigma)**2 + tf.math.log(sigma)
    nll = tf.reduce_mean(nll)
    return nll
