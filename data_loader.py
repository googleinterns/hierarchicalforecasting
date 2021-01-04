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

NUM_TIME_STEPS = 1000


class Favorita:
    def __init__(self):
        self.gen_data()
        self.compute_weights()

    def gen_data(self):
        global NUM_TIME_STEPS

        try:
            with open('synthetic_data.pkl', 'rb') as fin:
                print('Loading generated synthetic data ...')
                self.tree, self.ts_data, self.global_cont_feats = \
                    pickle.load(fin)
                self.num_ts = self.tree.num_nodes
        except FileNotFoundError:
            '''
            From https://stats.stackexchange.com/questions/125946/generate-a-time-series-comprising-seasonal-trend-and-remainder-components-in-r
            '''
            TS = []
            d = flags.node_emb_dim
            T = 1000
            p = 10
            for i in range(d):
                gammas = [np.random.randn() for i in range(p)]
                mu = 0
                beta = 0

                ts = []
                for j in range(T):
                    gamma = -np.sum(gammas[-p+1:]) + np.random.randn() * 0.1
                    gammas.append(gamma)
                    mu = mu + beta + np.random.randn() * 0.1
                    beta += np.random.randn() * 0.0001

                    y = mu + gamma + np.random.randn() * 0.1
                    ts.append(y)
                
                TS.append(ts)

            TS = np.array(TS).T
            
            n_1 = 10
            n_2 = 10

            # root_node = np.random.randn(1, d) * 0.1
            # middle_nodes = root_node + np.random.randn(n_1, d) * 0.1
            # reps = np.repeat(middle_nodes, n_2, axis=0)
            # leaf_nodes = reps + np.random.randn(n_1 * n_2, d) * 0.03

            root_node = [np.random.dirichlet(np.ones(d) / d * 10)]
            middle_nodes = []
            leaf_nodes = []

            for i in range(n_1):
                middle_node = np.random.dirichlet(root_node[0] * 7)
                middle_nodes.append(middle_node)
                for j in range(n_2):
                    leaf_node = np.random.dirichlet(middle_node * 3)
                    leaf_nodes.append(leaf_node)

            all_nodes = np.concatenate([root_node, middle_nodes, leaf_nodes])

            node_strs = ['r']
            for i in range(n_1):
                node_strs.append(f'{i}')
            
            for i in range(n_1):
                for j in range(n_2):
                    node_strs.append(f'{i}_{j}')
            
            assert(len(node_strs) == all_nodes.shape[0])

            self.tree = Tree()
            for nid in node_strs[1:]:  # excluding the root node
                self.tree.insert_seq(nid)
            self.tree.precompute()

            self.num_ts = self.tree.num_nodes
            print('NUM TS', self.num_ts)
            
            perm = [None for _ in node_strs]
            for i, node_str in enumerate(node_strs):
                nid = self.tree.node_id[node_str]
                assert(nid is not None)
                perm[nid] = i
            
            perm_nodes = all_nodes[perm]

            self.ts_data = TS @ perm_nodes.T
            print(self.ts_data.shape)
            self.global_cont_feats = np.asarray([i % p for i in range(NUM_TIME_STEPS)])
            self.global_cont_feats = self.global_cont_feats.reshape((-1, 1))

            with open('synthetic_data.pkl', 'wb') as fout:
                pickle.dump((self.tree, self.ts_data, self.global_cont_feats), fout)
    
    def compute_weights(self):
        levels = self.tree.levels
        self.w = np.ones(self.num_ts)
        for _, level in levels.items():
            self.w[level] /= len(level)
        self.w /= len(levels)
        assert(np.abs(np.sum(self.w) - 1.0) <= 1e-5)

    def train_gen(self):
        cont_len = flags.cont_len
        all_idx = np.arange(NUM_TIME_STEPS - 3 * cont_len)

        if flags.data_fraction < 1.0:
            start_idx = int((1 - flags.data_fraction) * NUM_TIME_STEPS)
            all_idx = all_idx[start_idx:]

        perm = np.random.permutation(all_idx)
        leaves = np.where(self.tree.leaf_vector)[0]
        idx = leaves
        # idx = np.arange(self.num_ts)

        for i in perm:
            sub_feat_cont = self.global_cont_feats[i:i+2*cont_len]
            sub_ts = self.ts_data[i:i+2*cont_len, idx]
            yield sub_feat_cont, sub_ts, idx  # t x *

    def val_gen(self):
        cont_len = flags.cont_len
        start = NUM_TIME_STEPS - 2 * cont_len

        sub_feat_cont = self.global_cont_feats[start:]
        j = np.arange(self.num_ts)
        sub_ts = self.ts_data[start:]
        yield sub_feat_cont, sub_ts, j  # t x *

    # def default_gen(self):
    #     cont_len = flags.cont_len
    #     tot_len = NUM_TIME_STEPS
    #     for i in range(0, tot_len - cont_len):
    #         a = i
    #         b = i + cont_len + 1
    #         sub_ts = self.ts_data[a:b]
    #         sub_feat_cont = self.global_cont_feats[a:b]
    #         sub_feat_cat = tuple(
    #             feat[a:b] for feat in self.global_cat_feats
    #         )
    #         j = np.arange(self.num_ts)
    #         yield (sub_feat_cont, sub_feat_cat), sub_ts, j  # t x *

    def tf_dataset(self, train):
        if train==True:
            dataset = tf.data.Dataset.from_generator(
                self.train_gen,
                (
                    tf.float32,  # feats
                    tf.float32,  # y_obs
                    tf.int32,  # id
                )
            )
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        elif train==False:
            dataset = tf.data.Dataset.from_generator(
                self.val_gen,
                (
                    tf.float32,  # feats
                    tf.float32,  # y_obs
                    tf.int32,  # id
                )
            )
        # elif train is None:
        #     dataset = tf.data.Dataset.from_generator(
        #         self.default_gen,
        #         (
        #             (tf.float32, (tf.int32, tf.int32, tf.int32)),  # feats
        #             tf.float32,  # y_obs
        #             tf.int32,  # id
        #         )
        #     )
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

