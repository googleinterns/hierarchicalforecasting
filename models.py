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

        self.lstm = layers.LSTM(flags.local_lstm_hidden,
                            return_sequences=False, time_major=True)
        self.emb = layers.Embedding(input_dim=self.num_ts,
                                    output_dim=flags.node_emb_dim)
        self.feat_emb = layers.Dense(flags.local_lstm_hidden, activation='tanh')
        self.dense = layers.Dense(1)

    @tf.function
    def call(self, feats, y_prev):
        '''
        feats: t x d
        y_prev: t x n
        '''

        # Computing local effects
        y_feats = tf.expand_dims(y_prev, -1)  # t x n x 1
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        stat_feats = tf.repeat(stat_feats, repeats=self.num_ts, axis=1)  # t x n x d
        
        idx = tf.range(self.num_ts)  # n
        idx = tf.expand_dims(idx, axis=0)  # 1 x n
        idx = tf.repeat(idx, repeats=self.time_steps, axis=0)  # t x n
        node_emb = self.emb(idx)  # t x n x e

        local_feats = tf.concat([y_feats, stat_feats, node_emb], axis=-1)  # t x n x D'
        local_feats = self.feat_emb(local_feats)  # t x n x e
        outputs = self.lstm(local_feats)  # n x h
        local_effect = self.dense(outputs)  # n x 1
        local_effect = tf.squeeze(local_effect, axis=-1)
        
        final_output = tf.math.softplus(local_effect)  # n

        return final_output
    
    @tf.function
    def train_step(self, feats, y_obs, optimizer):
        with tf.GradientTape() as tape:
            pred = self(feats[1:], y_obs[:-1])
            loss = tf.math.square(pred - y_obs[-1])

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
            pred = np.clip(pred.numpy(), a_min=0, a_max=None).reshape((1, -1))
            pred_path = np.concatenate([pred_path, pred], axis=0)
        
        return pred_path[-pred_hor:]

    def eval(self, dataset, level_dict):
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

            return_dict = {}
            rmses = []
            for d in level_dict:
                sub_mean = np.mean(rmse[level_dict[d]])
                rmses.append(sub_mean)
                return_dict[f'test/rmse@{d}'] = sub_mean

            return_dict['test/rmse'] = np.mean(rmses)

        np.save('notebooks/evals.npy', y_pred)
        return return_dict


@tf.function
def gaussian_nll(mean, sigma, y_obs):
    '''
    mean, sigma: t x n
    y_obs: t x n
    '''
    nll = 0.5 * ((y_obs - mean) / sigma)**2 + tf.math.log(sigma)
    nll = tf.reduce_mean(nll)
    return nll
