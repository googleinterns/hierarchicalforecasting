import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
import global_flags

from absl import app
from tqdm import tqdm

flags = global_flags.FLAGS

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class Data:
    def __init__(self):
        self.read_data()
        self.transform_data()
        self.compute_emb_matrix()
        self.compute_nmf()
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

    def compute_emb_matrix(self):
        """Create scaled embedding matrix."""
        tot_len = self.T
        num_data = tot_len - (flags.val_windows + flags.test_windows
                            ) * flags.pred_len - 2 * flags.hist_len
        yts = self.ts_data.transpose()[:, 0:num_data]
        mu = np.mean(yts, axis=1)
        sig = np.std(yts, axis=1)
        init_emb = np.random.normal(size=[self.num_ts, flags.node_emb_dim]).astype(
            np.float32)
        init_emb = init_emb * sig[:, None] + mu[:, None]
        self.tree.init_emb = init_emb

    def transform_data(self):
        """Compute the mean of each node."""
        tot_len = self.T
        num_data = tot_len - (flags.val_windows + flags.test_windows - 1
                            ) * flags.pred_len - flags.hist_len
        leaf_mat = self.tree.leaf_matrix.T
        num_leaf = np.sum(leaf_mat, axis=0, keepdims=True)
        self.ts_data = self.ts_data / num_leaf

        yts = self.ts_data.transpose()[:, :num_data]
        self.scalar = StandardScaler(mean=yts.mean(), std=yts.std())
        self.ts_data = self.scalar.transform(self.ts_data)

    def inverse_transform(self, pred):
        return self.scalar.inverse_transform(pred)
    
    def compute_nmf(self):

        #def fast_recursive_nmf(ymat: np.array, r: int):
        """Fast recursive NMF.

        Args:
            ymat: m X n matrix, need to find extreme columns for NMF
            r: rank of NMF

        Returns:
            A three tuple (tall factor, short factor, id's of columns selected).
        """
        norm_mat = self.ts_data.copy()
        num_data = self.T - (flags.val_windows + flags.test_windows - 1
                            ) * flags.pred_len - flags.hist_len
        norm_mat = norm_mat[:num_data]
        norm_mat = norm_mat / np.sum(norm_mat, axis=0, keepdims=True)

        self.nmf_cols = []
        m = norm_mat.shape[0]

        for _ in range(flags.nmf_rank):
            l2_norms = np.linalg.norm(norm_mat, axis=0)
            jmax = np.argmax(l2_norms)
            self.nmf_cols.append(jmax)
            pro_mat = np.eye(m) - np.dot(
                norm_mat[:, [jmax]], norm_mat[:, [jmax]].transpose()) / np.square(l2_norms[jmax])
            norm_mat = np.dot(pro_mat, norm_mat)
        self.nmf_ts = self.ts_data[:, self.nmf_cols]
        print('*' * 10, 'Selected columns from NMF', self.nmf_cols)
        # amat = np.linalg.lstsq(wmat, self.ts_data)[0]

    def train_gen(self):
        hist_len = flags.hist_len
        pred_len = flags.pred_len
        tot_len = self.T

        num_data = \
            tot_len - (flags.val_windows + flags.test_windows) * flags.pred_len \
                - 2 * flags.hist_len
        perm = np.random.permutation(num_data)

        print('Train')
        print('Data start:', 0, 'Data end:', num_data + 2*flags.hist_len)

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
            nmf_ts = self.nmf_ts[i:i+hist_len]
            yield (sub_feat_cont, sub_feat_cat), sub_ts, nmf_ts, j  # t x *

    def val_gen(self):
        hist_len = flags.hist_len
        tot_len = self.T
        pred_len = flags.pred_len

        start_idx = \
            tot_len - (flags.val_windows + flags.test_windows) * flags.pred_len \
                - flags.hist_len
        end_idx = tot_len - (flags.test_windows + 1) * flags.pred_len - flags.hist_len

        print('Val')
        print('Data start:', start_idx + flags.hist_len, 'Data end:', end_idx + flags.hist_len + flags.pred_len)

        for i in range(start_idx, end_idx+1, pred_len):
            sub_ts = self.ts_data[i:i+hist_len+pred_len]
            nmf_ts = self.nmf_ts[i:i+hist_len]
            sub_feat_cont = self.global_cont_feats[i:i+hist_len+pred_len]
            sub_feat_cat = tuple(
                feat[i:i+hist_len+pred_len] for feat in self.global_cat_feats
            )
            j = np.arange(self.num_ts)
            yield (sub_feat_cont, sub_feat_cat), sub_ts, nmf_ts, j  # t x *
    
    def test_gen(self):
        hist_len = flags.hist_len
        tot_len = self.T
        pred_len = flags.pred_len

        start_idx = \
            tot_len - flags.test_windows * flags.pred_len \
                - flags.hist_len
        end_idx = tot_len - flags.pred_len - flags.hist_len

        print('Test')
        print('Data start:', start_idx + flags.hist_len, 'Data end:', end_idx + flags.hist_len + flags.pred_len)

        for i in range(start_idx, end_idx+1, pred_len):
            sub_ts = self.ts_data[i:i+hist_len+pred_len]
            nmf_ts = self.nmf_ts[i:i+hist_len]
            sub_feat_cont = self.global_cont_feats[i:i+hist_len+pred_len]
            sub_feat_cat = tuple(
                feat[i:i+hist_len+pred_len] for feat in self.global_cat_feats
            )
            j = np.arange(self.num_ts)
            yield (sub_feat_cont, sub_feat_cat), sub_ts, nmf_ts, j  # t x *

    def tf_dataset(self, mode):
        if mode == 'train':
            gen_fn = self.train_gen
        elif mode == 'val':
            gen_fn = self.val_gen
        elif mode == 'test':
            gen_fn = self.test_gen

        num_cat_feats = len(self.global_cat_dims)
        output_type = tuple([tf.int32] * num_cat_feats)
        dataset = tf.data.Dataset.from_generator(
            gen_fn,
            (
                (tf.float32, output_type),  # feats
                tf.float32,  # y_obs
                tf.float32,  # nmf_ts
                tf.int32,  # id
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

def main(_):
    data = Data()
    for m in ['train', 'val', 'test']:
        dataset = data.tf_dataset(m)
        for d in dataset:
            break
    # print(data.ts_data)
    # idx = data.tree.leaf_vector.astype(np.bool)
    # diff = data.ts_data[:, 0] - np.mean(data.ts_data[:, idx], axis=1)
    # diff = np.abs(diff)
    # print(np.sum(diff))
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

