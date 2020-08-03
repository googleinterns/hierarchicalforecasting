import tensorflow as tf
import numpy as np
import global_flags
import sys

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS
MAX_FEAT_EMB_DIM = 50


class SimpleRNN(keras.Model):
    def __init__(self, num_ts, train_weights, cat_dims, leaf_matrix=None):
        print('INIT SRNN')
        super().__init__()

        self.num_ts = num_ts
        self.time_steps = flags.cont_len
        self.train_weights = tf.convert_to_tensor(train_weights, dtype=tf.float32)
        self.cat_dims = cat_dims
        
        self.leaf_matrix = leaf_matrix
        if leaf_matrix is not None:
            num_leaves = np.sum(leaf_matrix, axis=1, keepdims=True)
            self.leaf_matrix = leaf_matrix / num_leaves
        # self.leaf_matrix = np.identity(3060, dtype=np.float32)
        print(self.leaf_matrix)

        self.node_emb = layers.Embedding(input_dim=self.num_ts,
                                    output_dim=flags.node_emb_dim)
        self.embs = [
            layers.Embedding(input_dim=dim, output_dim=min(MAX_FEAT_EMB_DIM, (dim+1)//2))
            for dim in self.cat_dims
        ]

        self.lstm = layers.LSTM(flags.local_lstm_hidden,
                            return_sequences=False, time_major=True)
        
        self.local_loading = layers.Dense(flags.local_lstm_hidden, use_bias=False)
        self.bias = tf.Variable(
            initial_value=np.zeros((self.num_ts)),
            trainable=True, dtype=tf.float32)

        if flags.use_global_model:
            self.lstm_global = layers.LSTM(flags.global_lstm_hidden,
                            return_sequences=False, time_major=True)
            self.global_loading = layers.Dense(flags.global_lstm_hidden, use_bias=False)
    
    @tf.function
    def get_node_emb(self):
        print('CALL A')
        idx = tf.range(self.num_ts)  # n
        node_emb = self.node_emb(idx) # n x e
        if self.leaf_matrix is not None:
            node_emb = tf.matmul(self.leaf_matrix, node_emb)
        return node_emb

    @tf.function
    def assemble_feats(self, feats):

        feats_cont = feats[0]  # t x d
        feats_cat = feats[1]  # [t, t]
        feats_emb = [
            emb(feat) for emb, feat in zip(self.embs, feats_cat)  # t x e
        ]
        all_feats = feats_emb + [feats_cont]  # [t x *]
        all_feats = tf.concat(all_feats, axis=-1)
        return all_feats
    
    @tf.function
    def get_local(self, feats, y_prev):
        y_feats = tf.expand_dims(y_prev, -1)  # t x n x 1
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        stat_feats = tf.repeat(stat_feats, repeats=self.num_ts, axis=1)  # t x n x d
        
        node_emb = self.get_node_emb()

        local_feats = tf.concat([y_feats, stat_feats], axis=-1)  # t x n x D'
        # removed emb from local feats

        outputs = self.lstm(local_feats)  # n x h
        loadings = self.local_loading(node_emb)  # n x h

        local_effect = tf.reduce_sum(outputs * loadings, axis=1)  # n
        local_effect = local_effect + self.bias  # n
        return local_effect
    
    @tf.function
    def get_global(self, feats, y_prev):
        ''' loadings: n x h '''
        y_feats = tf.expand_dims(y_prev, 1)  # t x 1 x n
        stat_feats = tf.expand_dims(feats, axis=1)  # t x 1 x d
        global_feats = tf.concat([y_feats, stat_feats], axis=-1)  # t x 1 x D
        outputs = self.lstm_global(global_feats)  # 1 x h

        node_emb = self.get_node_emb()  # n x e
        loadings = self.global_loading(node_emb)  # n x h
        global_effect = tf.matmul(loadings, tf.transpose(outputs))  # n x 1
        global_effect = tf.squeeze(global_effect, 1)
        return global_effect

    @tf.function
    def call(self, feats, y_prev):
        '''
        feats: t x d
        y_prev: t x n
        '''
        feats = self.assemble_feats(feats)  # t x d

        # Computing local effects
        local_effect = self.get_local(feats, y_prev)
        final_output = local_effect  # n
        if flags.use_global_model:
            final_output = final_output + self.get_global(feats, y_prev)
        # final_output = tf.math.softplus(final_output)  # n
        return final_output
    
    @tf.function
    def train_step(self, feats, y_obs, optimizer):
        with tf.GradientTape() as tape:
            pred = self(
                self.get_sub_feats(feats, 1),
                y_obs[:-1]
            )
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
            pred = self(
                self.get_sub_feats(feats, i, i+cont_len),
                pred_path[i:i+cont_len]
            )
            pred = np.clip(pred.numpy(), a_min=0, a_max=None).reshape((1, -1))
            pred_path = np.concatenate([pred_path, pred], axis=0)
        
        return pred_path[-pred_hor:]

    def eval(self, dataset, level_dict):
        cont_len = flags.cont_len
        pred_hor = flags.pred_hor

        for feats, y_obs in dataset:
            y_obs = y_obs.numpy()
            
            assert(y_obs.shape[0] == cont_len + pred_hor)
            assert(feats[0].numpy().shape[0] == cont_len + pred_hor)

            y_pred = self.forecast(
                self.get_sub_feats(feats, 1),
                y_obs[:cont_len]
            )
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

    @staticmethod
    def get_sub_feats(feats, start, end=None):
        return (
            feats[0][start:end],
            [feat[start:end] for feat in feats[1]],
        )

class HierarchicalSimpleRNN(SimpleRNN):
    def __init__(self, num_ts, train_weights, cat_dims, leaf_matrix):
        print('INIT HSRNN')
        super(HierarchicalSimpleRNN, self).__init__(num_ts, train_weights, cat_dims)
        # num_leaves = np.sum(leaf_matrix, axis=1, keepdims=True)
        # self.leaf_matrix = leaf_matrix / num_leaves
        self.leaf_matrix = np.identity(3060, dtype=np.float32)
        print(self.leaf_matrix)

    # @tf.function
    # def get_node_emb(self):
    #     print('CALL B')
    #     idx = tf.range(self.num_ts)  # n
    #     node_emb = self.node_emb(idx) # n x e
    #     # node_emb = tf.matmul(self.leaf_matrix, node_emb)
    #     return node_emb
