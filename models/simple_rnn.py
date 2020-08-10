import tensorflow as tf
import numpy as np
import global_flags
import sys

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS
MAX_FEAT_EMB_DIM = 50


class FixedRNN(keras.Model):
    def __init__(self, num_ts, cat_dims, leaf_matrix=None):
        super().__init__()

        self.num_ts = num_ts
        self.time_steps = flags.cont_len
        self.cat_dims = cat_dims
        
        self.leaf_matrix = leaf_matrix
        if leaf_matrix is not None:
            num_leaves = np.sum(leaf_matrix, axis=1, keepdims=True)
            self.leaf_matrix = leaf_matrix / num_leaves
        # self.leaf_matrix = np.identity(3060, dtype=np.float32)

        self.node_emb = layers.Embedding(input_dim=self.num_ts,
            output_dim=flags.node_emb_dim,
            name='node_embed')
        

        self.embs = [
            layers.Embedding(input_dim=dim, output_dim=min(MAX_FEAT_EMB_DIM, (dim+1)//2))
            for dim in self.cat_dims
        ]

        self.lstm = layers.LSTM(flags.local_lstm_hidden,
                            return_sequences=False, time_major=True)
        
        self.local_loading = layers.Dense(flags.local_lstm_hidden, use_bias=False)
    
    def get_node_emb(self, nid):
        self.node_emb(np.asarray([0], dtype=np.int32))  # creates the emb matrix
        embs = self.node_emb.trainable_variables[0]
        if flags.hierarchy == 'additive':
            embs = tf.matmul(self.leaf_matrix, embs)
        node_emb = tf.nn.embedding_lookup(embs, nid)
        return node_emb

    def assemble_feats(self, feats):
        feats_cont = feats[0]  # t x d
        feats_cat = feats[1]  # [t, t]
        feats_emb = [
            emb(feat) for emb, feat in zip(self.embs, feats_cat)  # t x e
        ]
        all_feats = feats_emb + [feats_cont]  # [t x *]
        all_feats = tf.concat(all_feats, axis=-1)  # t x d
        return all_feats
    
    def get_local(self, feats, y_prev, nid):
        y_prev = tf.expand_dims(y_prev, -1)  # t x b x 1
        feats = tf.expand_dims(feats, 1)  # t x 1 x d
        feats = tf.repeat(feats, repeats=nid.shape[0], axis=1)  # t x b x d
        local_feats = tf.concat([y_prev, feats], axis=-1)  # t x b x D'

        node_emb = self.get_node_emb(nid)
        loadings = self.local_loading(node_emb)  # b x h

        outputs = self.lstm(local_feats)  # b x h

        local_effect = tf.reduce_sum(outputs * loadings, axis=1)  # b
        return local_effect

    @tf.function
    def call(self, feats, y_prev, nid):
        '''
        feats: t x d, t
        y_prev: t x b
        nid: b
        sw: b
        '''
        feats = self.assemble_feats(feats)  # t x d

        # Computing local effects
        local_effect = self.get_local(feats, y_prev, nid)  # b
        # final_output = tf.math.softplus(final_output)  # n
        return local_effect
    
    @tf.function
    def train_step(self, feats, y_obs, nid, sw, optimizer):
        '''
        feats:  b x t x d, b x t
        y_obs:  b x t
        nid: b
        sw: b
        ''' 
        with tf.GradientTape() as tape:
            pred = self(
                self.get_sub_feats(feats, 1),
                y_obs[:-1, :], nid
            )  # b
            # loss = tf.reduce_sum(tf.abs(pred - y_obs[-1, :]) * sw)
            loss = tf.reduce_mean(tf.abs(pred - y_obs[-1]))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        print(self.trainable_variables)

        return loss
    
    def forecast(self, feats, y_prev, nid):
        '''
        feats: (t+p-1) x n
        y_prev: t x n
        '''
        cont_len = flags.cont_len
        pred_hor = flags.pred_hor
        pred_path = y_prev.numpy()
        
        for i in range(pred_hor):
            pred = self(
                self.get_sub_feats(feats, i, i+cont_len),
                pred_path[i:i+cont_len],
                nid
            )
            pred = np.clip(pred.numpy(), a_min=0, a_max=None).reshape((1, -1))
            pred_path = np.concatenate([pred_path, pred], axis=0)
        
        return pred_path[-pred_hor:]

    def eval(self, dataset, level_dict):
        cont_len = flags.cont_len
        pred_hor = flags.pred_hor

        for feats, y_obs, nid in dataset:
            assert(y_obs.numpy().shape[0] == cont_len + pred_hor)
            assert(feats[0].numpy().shape[0] == cont_len + pred_hor)

            y_pred = self.forecast(
                self.get_sub_feats(feats, 1),
                y_obs[:cont_len],
                nid
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
