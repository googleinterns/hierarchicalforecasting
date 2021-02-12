import tensorflow as tf
import numpy as np
import global_flags
import sys
import pandas as pd
from tabulate import tabulate

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS
MAX_FEAT_EMB_DIM = 50
EPS = 1e-7


class FixedRNN(keras.Model):
    def __init__(self, num_ts, cat_dims, tree):
        super().__init__()

        self.num_ts = num_ts
        self.cat_dims = cat_dims
        
        self.tree = tree

        assert(flags.fixed_lstm_hidden == flags.node_emb_dim)

        self.node_emb = tf.Variable(
            np.random.uniform(size=[self.num_ts, flags.node_emb_dim]).astype(np.float32),
            name='node_emb'
        )

        self.cat_feat_embs = [
            layers.Embedding(input_dim=dim, output_dim=min(MAX_FEAT_EMB_DIM, (dim+1)//2))
            for dim in self.cat_dims
        ]

        self.encoder = layers.LSTM(flags.fixed_lstm_hidden,
                                return_state=True, time_major=True)
        self.decoder = layers.LSTM(flags.fixed_lstm_hidden,
                                return_sequences=True, time_major=True)
    
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
    
    def get_fixed(self, feats, y_prev, nid):
        y_prev = tf.expand_dims(y_prev, -1)  # t/2 x b x 1

        node_emb = self.get_node_emb(nid)  # b x h
        
        feats = tf.expand_dims(feats, 1)  # t x 1 x d
        feats = tf.repeat(feats, repeats=nid.shape[0], axis=1)  # t x b x d

        feats_prev = feats[:flags.hist_len]  # t/2 x b x d
        feats_futr = feats[flags.hist_len:]  # t/2 x b x d

        enc_inp = tf.concat([y_prev, feats_prev], axis=-1)  # t/2 x b x D'

        loadings = tf.expand_dims(node_emb, 0)  # 1 x b x h

        _, h, c = self.encoder(inputs=enc_inp)  # b x h
        output = self.decoder(inputs=feats_futr, initial_state=(h, c))  # t x b x h

        fixed_effect = tf.reduce_sum(output * loadings, axis=-1)  # t x b
        return fixed_effect

    def l_p_norm_reg(self, p):
        A = self.tree.adj_matrix  # n x n
        A = np.expand_dims(A, axis=0)  # 1 x n x n

        embs = self.get_transformed_emb()  # n x e
        embs = tf.transpose(embs)  # e x n
        emb_1 = tf.expand_dims(embs, axis=-1)  # e x n x 1
        emb_2 = tf.expand_dims(embs, axis=-2)  # e x 1 x n
        
        edge_diff = emb_1 * A - emb_2 * A
        if p==1:
            edge_diff = tf.abs(edge_diff)  # L1 regularization
        elif p==2:
            edge_diff = edge_diff * edge_diff  # L2 regularization
        reg = flags.reg_weight * tf.reduce_mean(edge_diff)

        return reg

    def regularizers(self):
        if flags.reg_type == 'l1':
            reg = self.l_p_norm_reg(p=1)
        elif flags.reg_type == 'l2':
            reg = self.l_p_norm_reg(p=2)
        elif flags.reg_type is None:
            reg = 0
        else:
            raise ValueError(f'Unknown regularization of type {flags.reg_type}')

        return reg

    @tf.function
    def call(self, feats, y_prev, nid):
        '''
        feats: t x d, t
        y_prev: t x b
        nid: b
        sw: b
        '''
        feats = self.assemble_feats(feats)  # t x d

        # Computing fixed effects
        fixed_effect = self.get_fixed(feats, y_prev, nid)  # t x b
        # final_output = tf.math.softplus(final_output)  # n
        return fixed_effect

    @tf.function
    def train_step(self, feats, y_obs, nid, optimizer):
        '''
        feats:  b x t x d, b x t
        y_obs:  b x t
        nid: b
        sw: b
        ''' 
        with tf.GradientTape() as tape:
            pred = self(feats, y_obs[:flags.hist_len], nid)  # t x 1
            mae = tf.abs(pred - y_obs[flags.hist_len:])  # t x 1
            mae = tf.reduce_mean(mae)
            loss = mae + self.regularizers()

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

        for feats, y_obs, nid in iterator:
            assert(y_obs.numpy().shape[0] == hist_len + pred_len)
            assert(feats[0].numpy().shape[0] == hist_len + pred_len)

            y_pred = self(feats, y_obs[:hist_len], nid)
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

        return df


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
