import tensorflow as tf
import numpy as np
import global_flags
import sys

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS

class SimpleRNN(keras.Model):
    def __init__(self, num_ts, train_weights):
        super(SimpleRNN, self).__init__()

        self.num_ts = num_ts
        self.time_steps = flags.cont_len
        self.train_weights = tf.convert_to_tensor(train_weights, dtype=tf.float32)

        self.lstm = layers.LSTM(flags.local_lstm_hidden,
                            return_sequences=False, time_major=True)
        self.emb = layers.Embedding(input_dim=self.num_ts,
                                    output_dim=flags.node_emb_dim)
        self.feat_emb = layers.Dense(flags.feat_dim_local, activation='tanh')
        self.dense = layers.Dense(1)
        
        self.scale = tf.Variable(
            initial_value=np.ones((self.num_ts)),
            trainable=True, dtype=tf.float32)
        self.bias = tf.Variable(
            initial_value=np.zeros((self.num_ts)),
            trainable=True, dtype=tf.float32)

        if flags.use_global_model:
            self.lstm_global = layers.LSTM(flags.global_lstm_hidden,
                            return_sequences=False, time_major=True)
            self.feat_emb_global = layers.Dense(flags.feat_dim_global, activation='tanh')
            self.dense_loading = layers.Dense(flags.global_lstm_hidden, use_bias=False)
    
    @tf.function
    def get_local(self, feats, y_prev):
        y_feats = tf.expand_dims(y_prev, -1)  # t x n x 1
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        stat_feats = tf.repeat(stat_feats, repeats=self.num_ts, axis=1)  # t x n x d
        
        idx = tf.range(self.num_ts)  # n
        raw_emb = self.emb(idx) # n x e
        emb = tf.expand_dims(raw_emb, axis=0)  # 1 x n x e
        emb = tf.repeat(emb, repeats=self.time_steps, axis=0)  # t x n x e

        local_feats = tf.concat([y_feats, stat_feats, emb], axis=-1)  # t x n x D'
        local_feats = self.feat_emb(local_feats)  # t x n x e
        outputs = self.lstm(local_feats)  # n x h
        local_effect = self.dense(outputs)  # n x 1
        local_effect = tf.squeeze(local_effect, axis=-1)
        local_effect = local_effect * self.scale + self.bias  # n
        return local_effect, raw_emb
    
    @tf.function
    def get_global(self, feats, y_prev, loadings):
        ''' loadings: n x h '''
        y_feats = tf.expand_dims(y_prev, 1)  # t x 1 x n
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        global_feats = tf.concat([y_feats, stat_feats], axis=-1)  # t x 1 x D
        global_feats = self.feat_emb_global(global_feats)
        outputs = self.lstm_global(global_feats)  # 1 x h
        global_effect = tf.matmul(loadings, tf.transpose(outputs))  # n x 1
        global_effect = tf.squeeze(global_effect, 1)
        return global_effect

    @tf.function
    def call(self, feats, y_prev):
        '''
        feats: t x d
        y_prev: t x n
        '''

        # Computing local effects
        local_effect, emb = self.get_local(feats, y_prev)
        final_output = local_effect  # n
        # final_output = 0
        if flags.use_global_model:
            loadings = self.dense_loading(emb)  # n x h
            final_output = final_output + self.get_global(feats, y_prev, loadings)
        # final_output = tf.math.softplus(final_output)  # n
        return final_output
    
    @tf.function
    def train_step(self, feats, y_obs, optimizer):
        with tf.GradientTape() as tape:
            pred = self(feats[1:], y_obs[:-1])
            loss = tf.reduce_sum(tf.abs(pred - y_obs[-1]) * self.train_weights)
            # loss = tf.reduce_mean(tf.abs(pred - y_obs[-1]))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        print(self.trainable_variables)

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
