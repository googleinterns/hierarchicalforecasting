import tensorflow as tf
import numpy as np
import global_flags
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


flags = global_flags.FLAGS

class ProbRNN(keras.Model):
    def __init__(self, num_ts, train_weights):
        super(ProbRNN, self).__init__()

        self.num_ts = num_ts
        self.time_steps = flags.cont_len
        self.train_weights = tf.convert_to_tensor(train_weights, dtype=tf.float32)

        self.lstm = layers.LSTM(flags.local_lstm_hidden,
                            return_sequences=False, time_major=True)
        self.lstm_var = layers.LSTM(flags.var_lstm_hidden,
                            return_sequences=False, time_major=True)
        self.emb = layers.Embedding(input_dim=self.num_ts,
                                    output_dim=flags.node_emb_dim)
        self.feat_emb = layers.Dense(flags.feat_dim_local, activation='tanh')
        self.dense_mean = layers.Dense(1)
        self.dense_var = layers.Dense(1)
        
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
    def get_emb(self):
        idx = tf.range(self.num_ts)  # n
        raw_emb = self.emb(idx) # n x e
        return raw_emb

    @tf.function
    def get_global(self, feats, y_prev):
        y_feats = tf.expand_dims(y_prev, 1)  # t x 1 x n
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        global_feats = tf.concat([y_feats, stat_feats], axis=-1)  # t x 1 x D
        global_feats = self.feat_emb_global(global_feats)
        outputs = self.lstm_global(global_feats)  # 1 x h

        emb = self.get_emb()  # n x e
        loadings = self.dense_loading(emb)  # n x h
        global_effect = tf.matmul(loadings, tf.transpose(outputs))  # n x 1
        global_effect = tf.squeeze(global_effect, 1)  # n
        return global_effect

    @tf.function
    def get_local(self, feats, y_prev):
         # Computing local effects
        y_feats = tf.expand_dims(y_prev, -1)  # t x n x 1
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        stat_feats = tf.repeat(stat_feats, repeats=self.num_ts, axis=1)  # t x n x d
        
        emb = self.get_emb()
        emb = tf.expand_dims(emb, axis=0)  # 1 x n x e
        emb = tf.repeat(emb, repeats=self.time_steps, axis=0)  # t x n x e

        local_feats = tf.concat([y_feats, stat_feats, emb], axis=-1)  # t x n x D'
        local_feats = self.feat_emb(local_feats)  # t x n x e
        
        mean_outputs = self.lstm(local_feats)  # n x h
        mean = self.dense_mean(mean_outputs)  # n x 1
        mean = tf.squeeze(mean, axis=-1)
        mean = mean * self.scale + self.bias  # n

        sig_outputs = self.lstm_var(local_feats)  # n x h
        sig = self.dense_var(sig_outputs)  # n x 1
        sig = tf.squeeze(sig, axis=-1)
        sig = sig * self.scale  # n

        return mean, sig

    @tf.function
    def call(self, feats, y_prev):
        '''
        feats: t x d
        y_prev: t x n
        '''
        mean, sig = self.get_local(feats, y_prev)

        if flags.use_global_model:
            global_effect = self.get_global(feats, y_prev)
            mean = mean + global_effect

        return mean, tf.math.softplus(sig)

    @tf.function
    def train_step(self, feats, y_obs, optimizer):
        with tf.GradientTape() as tape:
            mean, sig = self(feats[1:], y_obs[:-1])
            # loss = tf.reduce_sum(tf.abs(pred - y_obs[-1]) * self.train_weights)
            nll = gaussian_nll(mean, sig, y_obs[-1])
            loss = tf.reduce_sum(nll * self.train_weights)

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
            mean, sig = self.call(feats[i:i+cont_len],pred_path[i:i+cont_len])
            pred = mean + sig * tf.random.normal(sig.shape)
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
            
            preds = []
            print('Evaluating')
            for _ in tqdm(range(1000)):
                y_pred = self.forecast(feats[1:], y_obs[:cont_len])
                preds.append(y_pred)
            
            p = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
            p = np.array(p)
            quantiles = np.quantile(preds, p, axis=0)
            diff = np.expand_dims(y_obs[-pred_hor:], 0) - quantiles  # p x t x n
            q_lt_y = np.greater(diff, 0, dtype=np.float32)  # p x t x n
            spl = diff * q_lt_y * p.reshape((-1, 1, 1))
            spl += -diff * (1.0 - q_lt_y) * (1.0 - p.reshape((-1, 1, 1)))  # p x t x n

            mean = np.mean(spl, axis=0)  # t x n
            mean = np.mean(mean, axis=0)  # n

            return_dict = {}
            spls = []
            for d in level_dict:
                sub_mean = np.mean(mean[level_dict[d]])
                spls.append(sub_mean)
                return_dict[f'test/spl@{d}'] = sub_mean

            return_dict['test/spl'] = np.mean(spls)

        np.save('notebooks/evals.npy', preds)
        return return_dict


@tf.function
def gaussian_nll(mean, sigma, y_obs):
    '''
    mean, sigma: t x n
    y_obs: t x n
    '''
    nll = 0.5 * ((y_obs - mean) / sigma)**2 + tf.math.log(sigma)
    return nll
