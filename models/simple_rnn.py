import tensorflow as tf
import numpy as np
import global_flags
import sys
import pandas as pd
import pickle
from tabulate import tabulate

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS
MAX_FEAT_EMB_DIM = 5
EPS = 1e-7


class FixedRNN(keras.Model):
    def __init__(self, num_ts, cat_dims, tree):
        super().__init__()

        self.num_ts = num_ts
        self.cat_dims = cat_dims

        self.tree = tree
        init_mat = self.tree.init_emb.astype(np.float32)
        init_mat = self.tree.ancestor_matrix @ init_mat

        self.node_emb = tf.Variable(init_mat, name='node_emb')

        cat_emb_dims = [
            min(MAX_FEAT_EMB_DIM, (dim + 1) // 2) for dim in self.cat_dims
        ]
        print('Cat feat emb dims: %s', cat_emb_dims)
        self.cat_feat_embs = [
            layers.Embedding(input_dim=idim, output_dim=odim)
            for idim, odim in zip(self.cat_dims, cat_emb_dims)
        ]

        self.ar_encoder = layers.LSTM(flags.fixed_lstm_hidden,
                            return_state=True, time_major=True)
        self.ar_decoder = layers.LSTM(flags.fixed_lstm_hidden,
                            return_sequences=True, time_major=True)
        self.ar_output = layers.Dense(flags.hist_len, use_bias=True)

        self.emb_encoder = layers.LSTM(flags.fixed_lstm_hidden,
                            return_state=True, time_major=True)
        self.emb_decoder = layers.LSTM(flags.fixed_lstm_hidden,
                            return_sequences=True, time_major=True)
        self.emb_output = layers.Dense(flags.node_emb_dim, use_bias=True)


    def get_node_emb(self, nid):
        embs = self.node_emb
        node_emb = tf.nn.embedding_lookup(embs, nid)
        return node_emb

    def assemble_feats(self, feats):
        feats_cont = feats[0]  # t x d
        feats_cat = feats[1]  # [t, t]
        feats_emb = [
            emb(feat) for emb, feat in zip(self.cat_feat_embs, feats_cat)  # t x e
        ]
        all_feats = feats_emb + [feats_cont]  # [t x *]
        all_feats = tf.concat(all_feats, axis=-1)  # t x d
        return all_feats

    def emb_regularizer(self):
        if flags.emb_reg_weight > 0:
            ''' Leaves close to the mean leaf embedding '''
            # leaf_mat = self.tree.leaf_matrix
            # leaf_mat = leaf_mat / np.sum(leaf_mat, axis=1, keepdims=True)
            # assert(np.abs(np.sum(leaf_mat[0]) - 1.0) < 1e-7)
            # embs = self.node_emb
            # leaf_means = leaf_mat @ embs
            # diff = embs - leaf_means
            # reg = tf.reduce_sum(tf.square(diff))
            # return flags.emb_reg_weight * reg
            
            ''' Leaves close to each embedding '''
            leaf_mat = self.tree.leaf_matrix
            leaf_vector = self.tree.leaf_vector
            reg = 0.0

            for i in range(len(leaf_vector)):
                if leaf_vector[i] < 0.5:
                    leaves = leaf_mat[i]
                    idx = np.where(leaves > 0.5)[0]
                    
                    embA = self.get_node_emb(np.array([i]))  # 1 x d
                    sub_emb = self.get_node_emb(idx)  # l x d
                    diff = tf.square(embA - sub_emb)  # l x d
                    sub_reg = tf.reduce_sum(diff, axis=1)  # l
                    sub_reg = tf.reduce_mean(diff)
                    reg += sub_reg
            
            return reg * flags.emb_reg_weight

        return 0.0

    @tf.function
    def call(self, feats, y_prev, z, nid):
        '''
        feats: t x d, t
        y_prev: t/2 x b
        z: t/2 x r
        nid: b
        sw: b
        '''
        feats = self.assemble_feats(feats)  # t x d
        feats = tf.expand_dims(feats, 1)  # t x 1 x d
        node_emb = self.get_node_emb(nid)  # b x e
        
        feats_prev = feats[:flags.hist_len]  # t/2 x 1 x d
        feats_futr = feats[flags.hist_len:]  # t/2 x 1 x d

        z = tf.expand_dims(z, 1)  # t/2 x 1 x r
        feats_prev = tf.concat([feats_prev, z], axis=-1)  # t x d
        _, h, c = self.ar_encoder(inputs=feats_prev)  # 1 x h
        ar = self.ar_decoder(inputs=feats_futr, initial_state=(h, c))  # t/2 x 1 x h
        ar = self.ar_output(ar)  # t/2 x 1 x t/2

        ar = tf.squeeze(ar, axis=1)  # t/2 x t/2
        ar = tf.matmul(ar, y_prev)  # t/2 x b

        _, h, c = self.emb_encoder(inputs=feats_prev)  # 1 x h
        ew = self.emb_decoder(inputs=feats_futr, initial_state=(h, c))  # t/2 x 1 x h
        ew = self.emb_output(ew)  # t/2 x 1 x e
        ew = tf.squeeze(ew, axis=1)  # t/2 x e
        ew = tf.nn.tanh(ew)
        ew = tf.matmul(ew, node_emb, transpose_b=True)  # t/2 x b

        final_output = ar + ew

        return final_output

    @tf.function
    def train_step(self, feats, y_obs, z, nid, optimizer):
        '''
        feats:  b x t x d, b x t
        y_obs:  b x t
        nid: b
        sw: b
        '''
        with tf.GradientTape() as tape:
            pred = self(feats, y_obs[:flags.hist_len], z, nid)  # t x 1
            mae = tf.abs(pred - y_obs[flags.hist_len:])  # t x 1
            mae = tf.reduce_mean(mae)
            loss = mae + self.emb_regularizer()

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        print('# Parameters in model', np.sum([np.prod(v.shape) for v in self.trainable_variables]))
        # print(self.trainable_variables)
        # for v in self.trainable_variables:
        #     print(v.name)
        #     if v.name == 'node_emb':
        #         print(v)

        return loss, mae

    def eval(self, data, mode):
        iterator = data.tf_dataset(mode=mode)
        level_dict = data.tree.levels
        hist_len = flags.hist_len
        pred_len = flags.test_pred

        all_y_true = None
        all_y_pred = None

        def set_or_concat(A, B):
            if A is None:
                return B
            return np.concatenate((A, B), axis=0)

        for feats, y_obs, z, nid in iterator:
            assert(y_obs.numpy().shape[0] == hist_len + pred_len)
            assert(feats[0].numpy().shape[0] == hist_len + pred_len)

            y_pred = self(feats, y_obs[:hist_len], z, nid)
            test_loss = tf.abs(y_pred - y_obs[hist_len:])  # t x 1
            test_loss = tf.reduce_mean(test_loss).numpy()

            y_pred = data.inverse_transform(y_pred.numpy())
            y_pred = np.clip(y_pred, 0.0, np.inf)
                    # Assuming predictions are positive

            y_true = y_obs[hist_len:].numpy()
            y_true = data.inverse_transform(y_true)

            all_y_pred = set_or_concat(all_y_pred, y_pred)
            all_y_true = set_or_concat(all_y_true, y_true)

        results_list = []

        '''Compute metrics for all time series together'''
        results_dict = {}
        results_dict['level'] = 'all'
        for metric in METRICS:
            eval_fn = METRICS[metric]
            results_dict[metric] = eval_fn(all_y_pred, all_y_true)
        results_list.append(results_dict)

        '''Compute metrics for individual levels and their mean across levels'''
        mean_dict = {metric: [] for metric in METRICS}

        for d in level_dict:
            results_dict = {}
            sub_pred = all_y_pred[:, level_dict[d]]
            sub_true = all_y_true[:, level_dict[d]]
            for metric in METRICS:
                eval_fn = METRICS[metric]
                eval_val = eval_fn(sub_pred, sub_true)
                results_dict[metric] = eval_val
                mean_dict[metric].append(eval_val)
            results_dict['level'] = d
            results_list.append(results_dict)
        
        '''Compute the mean result of all the levels'''
        for metric in mean_dict:
            mean_dict[metric] = np.mean(mean_dict[metric])
        mean_dict['level'] = 'mean'
        results_list.append(mean_dict)
        
        df = pd.DataFrame(data=results_list)
        df.set_index('level', inplace=True)
        print('\n###', mode.upper())
        print(tabulate(df, headers='keys', tablefmt='psql'))
        print(f'Loss: {test_loss}')

        return df, (all_y_pred, all_y_true)


def mape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true).flatten()
    abs_val = np.abs(y_true).flatten()
    idx = np.where(abs_val > 0.1)
    mape = np.mean(abs_diff[idx]/abs_val[idx])
    return mape

def wape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_val = np.abs(y_true)
    wape = np.sum(abs_diff)/(np.sum(abs_val) + EPS)
    return wape

def smape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_mean = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(abs_diff/(abs_mean + EPS))
    return smape


METRICS = {'mape': mape, 'wape': wape, 'smape': smape}
