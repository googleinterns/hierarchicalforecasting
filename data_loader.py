import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
import global_flags

from absl import app
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, minmax_scale

flags = global_flags.FLAGS

NUM_TIME_STEPS = 1941 - 28  # Offsetting by 28 # Starts from 1
START_IDX = 1000

class M5Data:
    def __init__(self):
        self.read_data()
        self.compute_weights()
        if flags.model == 'fixed':
            self.variation_scaling_A()
        elif flags.model == 'random':
            self.variation_scaling_B()
        else:
            raise ValueError(f'Unknown model {flags.model}')

    def read_data(self):
        data_path = os.path.join(flags.m5dir, 'sales_train_evaluation.csv')
        feats_path = os.path.join(flags.m5dir, 'calendar.csv')
        pkl_path = os.path.join(flags.m5dir, 'data.pkl')

        try:
            with open(pkl_path, 'rb') as fin:
                print('Found pickle. Loading ...')
                self.tree, self.num_ts, self.ts_data, \
                    (self.global_cont_feats, self.global_cat_feats, self.global_cat_dims) \
                        = pickle.load(fin)
        except FileNotFoundError:
            print('Pickle not found. Reading from csv ...')
            df = pd.read_csv(data_path, ',')
            self.tree = Tree()

            col_name = 'item_id' # 'dept_id'
            for item_id in df[col_name]:
                self.tree.insert_seq(item_id)
            self.tree.precompute()
            
            self.num_ts = self.tree.num_nodes
            self.ts_data = np.zeros((self.num_ts, NUM_TIME_STEPS), dtype=np.float32)

            cols = df.columns
            node_str_idx = cols.get_loc(col_name) + 1
            d_1_idx = cols.get_loc('d_1') + 1
            d_n_idx = cols.get_loc(f'd_{NUM_TIME_STEPS}') + 1
            for row in tqdm(df.itertuples()):
                node_str = row[node_str_idx]
                a_ids = self.tree.get_ancestor_ids(node_str)
                ts = np.asarray(row[d_1_idx : d_n_idx+1])
                self.ts_data[a_ids] += ts
            self.ts_data = self.ts_data.T
            
            features = pd.read_csv(feats_path, ',')
            feats = np.asarray(
                features[['wday', 'month', 'snap_CA', 'snap_TX', 'snap_WI']]\
                    [:NUM_TIME_STEPS])
            feats = minmax_scale(feats)
            self.global_cont_feats = np.asarray(feats, dtype=np.float32)

            self.global_cat_feats = []
            self.global_cat_dims = []

            cat_feat_list = ['event_name_1', 'event_type_1']
            for cat_feat_name in cat_feat_list:
                feats = features[cat_feat_name][:NUM_TIME_STEPS].fillna('')
                feats = [[feat] for feat in feats]
                enc = OrdinalEncoder(dtype=np.int32)
                feats = enc.fit_transform(feats)
                self.global_cat_feats.append(np.asarray(feats, dtype=np.int32).ravel())
                self.global_cat_dims.append(len(enc.categories_[0]))

            feats = (self.global_cont_feats, self.global_cat_feats, self.global_cat_dims)
            with open(pkl_path, 'wb') as fout:
                pickle.dump((self.tree, self.num_ts, self.ts_data, feats),
                            fout)
    
    def compute_weights(self):
        levels = self.tree.levels
        self.w = np.ones(self.num_ts)
        for _, level in levels.items():
            self.w[level] /= len(level)
        self.w /= len(levels)
        assert(np.abs(np.sum(self.w) - 1.0) <= 1e-5)

    def mean_scaling(self):
        self.abs_means = np.mean(np.abs(self.ts_data), axis=0).reshape((1, -1))
        self.ts_data = self.ts_data / self.abs_means
    
    def variation_scaling_A(self):
        self.variations = self.ts_data[1:] - self.ts_data[:-1]
        self.variations = np.mean(self.variations**2, axis=0)
        self.variations = np.sqrt(self.variations).reshape((1, -1))
        self.ts_data = self.ts_data / self.variations
    
    def variation_scaling_B(self):
        self.variations = np.abs(self.ts_data[1:] - self.ts_data[:-1])
        self.variations = np.mean(self.variations, axis=0).reshape((1, -1))
        self.ts_data = self.ts_data / self.variations

    def train_gen(self):
        pred_hor = flags.pred_hor
        tot_len = NUM_TIME_STEPS

        num_data = tot_len - 3 * pred_hor
        perm = np.random.permutation(num_data)

        weights = self.w * (self.num_ts / flags.batch_size)

        for i in perm:
            sub_feat_cont = self.global_cont_feats[i:i+2*pred_hor]
            sub_feat_cat = tuple(
                feat[i:i+2*pred_hor] for feat in self.global_cat_feats
            )
            # j = np.random.choice(range(self.num_ts), size=flags.batch_size, replace=False)
            j = np.random.choice(range(self.num_ts), size=flags.batch_size, p=self.w)
            # j = np.random.permutation(3060)
            sub_ts = self.ts_data[i:i+2*pred_hor, j]
            yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *
        
    def val_gen(self):
        pred_hor = flags.pred_hor
        tot_len = NUM_TIME_STEPS
        start_idx = tot_len - 2 * pred_hor
        sub_ts = self.ts_data[start_idx:tot_len]
        sub_feat_cont = self.global_cont_feats[start_idx:tot_len]
        sub_feat_cat = tuple(
            feat[start_idx:tot_len] for feat in self.global_cat_feats
        )
        j = np.arange(self.num_ts)
        yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *

    def tf_dataset(self, train):
        if train:
            dataset = tf.data.Dataset.from_generator(
                self.train_gen,
                (
                    (tf.float32, (tf.int32, tf.int32)),  # feats
                    tf.float32,  # y_obs
                    tf.int32,  # id
                )
            )
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_generator(
                self.val_gen,
                (
                    (tf.float32, (tf.int32, tf.int32)),  # feats
                    tf.float32,  # y_obs
                    tf.int32,  # id
                )
            )
        return dataset


class Tree:
    root = 'r'

    def __init__(self):
        self.parent = {}
        self.children = {}
        self.node_id = {}
        self.id_node = {}

        self.insert_node(self.root, None)
    
    @property
    def num_nodes(self):
        return len(self.node_id)
    
    @staticmethod
    def get_ancestors(node_path):
        ancestors = []
        for i, c in enumerate(node_path):
            if c == '_':
                ancestors.append(node_path[:i])
        ancestors.append(node_path)
        return ancestors
    
    def insert_node(self, node_str, par_str):
        if node_str in self.node_id:
            return
        nid = len(self.node_id)
        self.node_id[node_str] = nid
        self.id_node[nid] = node_str
        self.parent[node_str] = par_str
        self.children[node_str] = []
        if par_str is not None:
            self.children[par_str].append(node_str)
    
    def insert_seq(self, node_path):
        ancestors = self.get_ancestors(node_path)
        par = self.root
        for a in ancestors:
            self.insert_node(a, par)
            par = a
    
    def get_ancestor_ids(self, node_str):
        ids = []
        node = node_str
        while node is not None:
            ids.append(self.node_id[node])
            node = self.parent[node]
        return ids
    
    def precompute(self):
        self.init_levels()
        self.init_matrix()
    
    def init_matrix(self):
        n = len(self.node_id)
        self.leaf_matrix = np.zeros((n, n), dtype=np.float32)
        self.ancestor_matrix = np.zeros((n, n), dtype=np.float32)
        self.adj_matrix = np.zeros((n, n), dtype=np.float32)

        self._dfs(self.root, [])
    
    def _dfs(self, node_str, ancestors):
        nid = self.node_id[node_str]
        if len(ancestors):
            par = ancestors[-1]
            self.adj_matrix[par, nid] = 1
            self.adj_matrix[nid, par] = 1
        ancestors = ancestors + [nid]
        self.ancestor_matrix[nid, ancestors] = 1
        if len(self.children[node_str]) == 0:  # leaf
            self.leaf_matrix[ancestors, nid] = 1
        else:
            for ch in self.children[node_str]:
                self._dfs(ch, ancestors)
    
    def init_levels(self):
        self.levels = {}
        self._levels_rec(self.root, 0)
    
    def _levels_rec(self, node_str, depth):
        if depth not in self.levels:
            self.levels[depth] = []
        self.levels[depth].append(self.node_id[node_str])
        for ch in self.children[node_str]:
            self._levels_rec(ch, depth+1)


def main(_):
    tree = Tree()
    tree.insert_seq('food_2_1')
    tree.insert_seq('food_2_2')
    tree.insert_seq('hobbies_1')
    tree.insert_seq('hobbies_2')

    tree.precompute()

    print(tree.parent)
    print(tree.children)
    print(tree.node_id)
    print(tree.id_node)

    print(tree.leaf_matrix)
    print(tree.adj_matrix)
    print(tree.ancestor_matrix)

    # data = M5Data()
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

