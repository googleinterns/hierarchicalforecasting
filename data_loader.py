import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
import global_flags

from absl import app
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, minmax_scale, StandardScaler

flags = global_flags.FLAGS

class Data:
    def __init__(self):
        self.read_data()
        self.transform_data()
        # self.compute_weights()

    def read_data(self):
        pkl_path = os.path.join('data', flags.dataset, 'data.pkl')
        with open(pkl_path, 'rb') as fin:
            print('Found pickle. Loading ...')
            self.tree, self.ts_data, \
                (self.global_cont_feats, self.global_cat_feats, self.global_cat_dims) \
                    = pickle.load(fin)
        self.T, self.num_ts = self.ts_data.shape

    def compute_weights(self):
        levels = self.tree.levels
        self.w = np.ones(self.num_ts)
        for _, level in levels.items():
            self.w[level] /= len(level)
        self.w /= len(levels)
        assert(np.abs(np.sum(self.w) - 1.0) <= 1e-5)

    def transform_data(self):
        # Compute the mean of each node
        leaf_mat = self.tree.leaf_matrix.T
        num_leaf = np.sum(leaf_mat, axis=0, keepdims=True)
        self.ts_data = self.ts_data / num_leaf
    
    def inverse_transform(self, pred):
        return pred

    def train_gen(self):
        hist_len = flags.hist_len
        pred_len = flags.train_pred
        tot_len = self.T

        num_data = \
            tot_len - (flags.val_windows + flags.test_windows) * flags.test_pred \
                - 2 * flags.hist_len
        perm = np.random.permutation(num_data)

        # weights = self.w * (self.num_ts / flags.batch_size)

        for i in perm:
            sub_feat_cont = self.global_cont_feats[i:i+hist_len+pred_len]
            sub_feat_cat = tuple(
                feat[i:i+hist_len+pred_len] for feat in self.global_cat_feats
            )
            # j = np.random.choice(range(self.num_ts), size=flags.batch_size, replace=False)
            # j = np.random.choice(range(self.num_ts), size=flags.batch_size, p=self.w)
            j = np.random.choice(range(self.num_ts), size=flags.batch_size)
            # j = np.random.permutation(3060)
            sub_ts = self.ts_data[i:i+hist_len+pred_len, j]
            yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *

    def val_gen(self):
        hist_len = flags.hist_len
        tot_len = self.T
        pred_len = flags.test_pred

        start_idx = \
            tot_len - (flags.val_windows + flags.test_windows) * flags.test_pred \
                - flags.hist_len
        end_idx = tot_len - (flags.test_windows + 1) * flags.test_pred - flags.hist_len
        for i in range(start_idx, end_idx, pred_len):
            sub_ts = self.ts_data[i:i+hist_len+pred_len]
            sub_feat_cont = self.global_cont_feats[i:i+hist_len+pred_len]
            sub_feat_cat = tuple(
                feat[i:i+hist_len+pred_len] for feat in self.global_cat_feats
            )
            j = np.arange(self.num_ts)
            yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *
    
    def test_gen(self):
        hist_len = flags.hist_len
        tot_len = self.T
        pred_len = flags.test_pred

        start_idx = \
            tot_len - flags.test_windows * flags.test_pred \
                - flags.hist_len
        end_idx = tot_len - flags.test_pred - flags.hist_len
        for i in range(start_idx, end_idx, pred_len):
            sub_ts = self.ts_data[i:i+hist_len+pred_len]
            sub_feat_cont = self.global_cont_feats[i:i+hist_len+pred_len]
            sub_feat_cat = tuple(
                feat[i:i+hist_len+pred_len] for feat in self.global_cat_feats
            )
            j = np.arange(self.num_ts)
            yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *

    def tf_dataset(self, mode):
        if mode == 'train':
            gen_fn = self.train_gen
        elif mode == 'val':
            gen_fn = self.val_gen
        elif mode == 'test':
            gen_fn = self.test_gen

        dataset = tf.data.Dataset.from_generator(
            gen_fn,
            (
                (tf.float32, (tf.int32, tf.int32)),  # feats
                tf.float32,  # y_obs
                tf.int32,  # id
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

def main(_):
    data = Data()
    print(data.ts_data)
    idx = data.tree.leaf_vector.astype(np.bool)
    diff = data.ts_data[:, 0] - np.mean(data.ts_data[:, idx], axis=1)
    diff = np.abs(diff)
    print(np.sum(diff))
    # print(data.ts_data.dtype, data.ts_data.shape)

    # dataset = data.tf_dataset(True)
    # for d in dataset:
    #     feats = d[0]
    #     y_obs = d[1]
    #     nid = d[2]
    #     sw = d[3]
    #     print(feats[0].shape)
    #     print(feats[1][0].shape, feats[1][1].shape)
    #     print(y_obs.shape)
    #     print(nid.shape, sw.shape)
    #     break

    # for d in tqdm(data.train_gen()):
    #     pass

    # dataset = data.tf_dataset(True)
    # for d in tqdm(dataset):
    #     d[0]

    # print(data.weights)
    # print(np.sum(data.weights))
    # print(data.weights.shape)


if __name__ == "__main__":
    app.run(main)

