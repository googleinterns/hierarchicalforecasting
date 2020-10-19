import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
import global_flags
import datetime as dt

from absl import app
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, minmax_scale
from datetime import datetime

flags = global_flags.FLAGS

START = "2016-06-29"
END = "2017-08-15"

FORMAT = "%Y-%m-%d"
START = datetime.strptime(START, FORMAT)
END = datetime.strptime(END, FORMAT)
DIFF = END-START

NUM_TIME_STEPS = DIFF.days + 1

class Favorita:
    def __init__(self):
        self.read_data()
        self.compute_weights()

    def read_data(self):
        global NUM_TIME_STEPS
        data_path = os.path.join(flags.favorita_dir, 'aggregate_sales10.csv')
        feats_path = os.path.join(flags.favorita_dir, 'holidays_events.csv')
        pkl_path = os.path.join(flags.favorita_dir, 'data.pkl')

        try:
            with open(pkl_path, 'rb') as fin:
                print('Found pickle. Loading ...')
                self.tree, self.num_ts, self.ts_data, \
                    (self.global_cont_feats, self.global_cat_feats, self.global_cat_dims) \
                        = pickle.load(fin)
        except FileNotFoundError:
            print('Pickle not found. Reading from csv ...')
            df = pd.read_csv(data_path, sep=',')
            df['date'] = pd.to_datetime(df.date)
            df.sort_values(by='date', inplace=True)

            self.tree = Tree()

            for item in df['item']:
                item_id = item // 100
                store_id = item % 100

                if store_id == 0:
                    continue
                node_str = f'{item_id}_{store_id}'
                self.tree.insert_seq(node_str)
            self.tree.precompute()
            
            self.num_ts = self.tree.num_nodes
            print('NUM TS', self.num_ts)
            self.ts_data = np.zeros((self.num_ts, NUM_TIME_STEPS), dtype=np.float32)
            print(self.ts_data.shape)

            for row in tqdm(df[['item', 'date', 'unit_sales']].itertuples()):
                item, date, sales = row[1], row[2], row[3]
                if (item % 100) == 0:
                    continue
                node_str = f'{item // 100}_{item % 100}'
                idx = (date - START).days
                a_ids = self.tree.get_ancestor_ids(node_str)
                self.ts_data[a_ids, idx] += sales

            # cols = df.columns
            # node_str_idx = cols.get_loc(col_name) + 1
            # d_1_idx = cols.get_loc('d_1') + 1
            # d_n_idx = cols.get_loc(f'd_{NUM_TIME_STEPS}') + 1
            # for row in tqdm(df.itertuples()):
            #     node_str = row[node_str_idx]
            #     a_ids = self.tree.get_ancestor_ids(node_str)
            #     ts = np.asarray(row[d_1_idx : d_n_idx+1])
            #     self.ts_data[a_ids] += ts
            self.ts_data = self.ts_data.T
            print('TS DATA', self.ts_data.shape)
            
            date_feats = []
            for i in range(NUM_TIME_STEPS):
                date = START + dt.timedelta(days=i)
                date_feats.append(date.isocalendar()[1:3])
            date_feats = minmax_scale(date_feats)
            self.global_cont_feats = np.asarray(date_feats, dtype=np.float32)
            print('CONT FEATS', self.global_cont_feats.shape)

            features = pd.read_csv(feats_path, sep=',')
            features['date'] = pd.to_datetime(features.date)
            features.sort_values(by='date', inplace=True)

            self.global_cat_feats = []
            self.global_cat_dims = []

            cat_feat_list = ['type', 'locale', 'locale_name']
            for cat_feat_name in cat_feat_list:
                feats = ['' for i in range(NUM_TIME_STEPS)]
                for row in features[['date', cat_feat_name]].itertuples():
                    date = row[1]
                    idx = (date - START).days
                    to_end = (END - date).days
                    # print(date, START, idx, to_end)
                    if idx < 0 or to_end < 0:
                        continue
                    event = row[2]
                    feats[idx] = event

                feats = [[feat] for feat in feats]
                enc = OrdinalEncoder(dtype=np.int32)
                feats = enc.fit_transform(feats)
                self.global_cat_feats.append(np.asarray(feats, dtype=np.int32).ravel())
                self.global_cat_dims.append(len(enc.categories_[0]))

            feats = (self.global_cont_feats, self.global_cat_feats, self.global_cat_dims)
            with open(pkl_path, 'wb') as fout:
                pickle.dump((self.tree, self.num_ts, self.ts_data, feats),
                            fout)

        if flags.load_alternate:
            with open('data/favorita/alternate_ts.pkl', 'rb') as fin:
                self.ts_data = pickle.load(fin)
            NUM_TIME_STEPS = len(self.ts_data)
            print('alternate_ts', NUM_TIME_STEPS)
    
    def compute_weights(self):
        levels = self.tree.levels
        self.w = np.ones(self.num_ts)
        for _, level in levels.items():
            self.w[level] /= len(level)
        self.w /= len(levels)
        assert(np.abs(np.sum(self.w) - 1.0) <= 1e-5)

    def train_gen(self):
        cont_len = flags.cont_len
        all_idx = np.arange(NUM_TIME_STEPS - 2 * cont_len)

        if flags.load_alternate:
            start_idx = flags.cont_len
            if flags.data_fraction < 1.0:
                start_idx = int((1 - flags.data_fraction) * NUM_TIME_STEPS)
            all_idx = all_idx[start_idx:]

        perm = np.random.permutation(all_idx)

        for i in perm:
            sub_feat_cont = self.global_cont_feats[i:i+cont_len+1]
            sub_feat_cat = tuple(
                feat[i:i+cont_len+1] for feat in self.global_cat_feats
            )
            j = np.random.choice(range(self.num_ts), size=flags.batch_size, p=self.w)
            sub_ts = self.ts_data[i:i+cont_len+1, j]
            yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *
        
    def val_gen(self):
        cont_len = flags.cont_len
        all_idx = np.arange(NUM_TIME_STEPS - 2 * cont_len, NUM_TIME_STEPS - cont_len - 1)

        for i in all_idx:
            sub_feat_cont = self.global_cont_feats[i:i+cont_len+1]
            sub_feat_cat = tuple(
                feat[i:i+cont_len+1] for feat in self.global_cat_feats
            )
            j = np.arange(self.num_ts)
            sub_ts = self.ts_data[i:i+cont_len+1]
            yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *
    
    def default_gen(self):
        cont_len = flags.cont_len
        tot_len = NUM_TIME_STEPS
        for i in range(0, tot_len - cont_len):
            a = i
            b = i + cont_len + 1
            sub_ts = self.ts_data[a:b]
            sub_feat_cont = self.global_cont_feats[a:b]
            sub_feat_cat = tuple(
                feat[a:b] for feat in self.global_cat_feats
            )
            j = np.arange(self.num_ts)
            yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *

    def tf_dataset(self, train):
        if train==True:
            dataset = tf.data.Dataset.from_generator(
                self.train_gen,
                (
                    (tf.float32, (tf.int32, tf.int32, tf.int32)),  # feats
                    tf.float32,  # y_obs
                    tf.int32,  # id
                )
            )
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        elif train==False:
            dataset = tf.data.Dataset.from_generator(
                self.val_gen,
                (
                    (tf.float32, (tf.int32, tf.int32, tf.int32)),  # feats
                    tf.float32,  # y_obs
                    tf.int32,  # id
                )
            )
        elif train is None:
            dataset = tf.data.Dataset.from_generator(
                self.default_gen,
                (
                    (tf.float32, (tf.int32, tf.int32, tf.int32)),  # feats
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
        self.parent_matrix = np.zeros((n, n), dtype=np.float32)
        self.leaf_vector = np.zeros(n, dtype=np.float32)

        self._dfs(self.root, [])
    
    def _dfs(self, node_str, ancestors):
        nid = self.node_id[node_str]
        if len(ancestors):
            par = ancestors[-1]
            self.adj_matrix[par, nid] = 1
            self.adj_matrix[nid, par] = 1
            self.parent_matrix[nid, par] = 1
        ancestors = ancestors + [nid]
        self.ancestor_matrix[nid, ancestors] = 1
        if len(self.children[node_str]) == 0:  # leaf
            self.leaf_matrix[ancestors, nid] = 1
            self.leaf_vector[nid] = 1
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
    # tree = Tree()
    # tree.insert_seq('food_2_1')
    # tree.insert_seq('food_2_2')
    # tree.insert_seq('hobbies_1')
    # tree.insert_seq('hobbies_2')

    # tree.precompute()

    # print(tree.parent)
    # print(tree.children)
    # print(tree.node_id)
    # print(tree.id_node)

    # print(tree.leaf_matrix)
    # print(tree.adj_matrix)
    # print(tree.ancestor_matrix)

    data = Favorita()
    print(data.ts_data.dtype, data.ts_data.shape)

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

