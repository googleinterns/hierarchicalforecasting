import numpy as np
import pandas as pd
import os
import pickle

from absl import app, flags
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, minmax_scale

flags.DEFINE_string('m5dir', './data/m5', 'Path to the m5 data directory')

flags = flags.FLAGS

NUM_TIME_STEPS = 1941  # Starts from 1

class M5Data:
    def __init__(self):
        data_path = os.path.join(flags.m5dir, 'sales_train_evaluation.csv')
        feats_path = os.path.join(flags.m5dir, 'calendar.csv')
        pkl_path = os.path.join(flags.m5dir, 'data.pkl')

        try:
            with open(pkl_path, 'rb') as fin:
                print('Found pickle. Loading ...')
                self.tree, self.num_ts, self.ts_data, self.feats = \
                    pickle.load(fin)
        except FileNotFoundError:
            print('Pickle not found. Reading from csv ...')
            df = pd.read_csv(data_path, ',')
            self.tree = Tree()
            for item_id in df['item_id']:
                self.tree.insert_seq(item_id)
            self.tree.init_leaf_count()
            
            self.num_ts = self.tree.num_nodes
            self.ts_data = np.zeros((self.num_ts, NUM_TIME_STEPS), dtype=np.float32)

            cols = df.columns
            node_str_idx = cols.get_loc('item_id') + 1
            d_1_idx = cols.get_loc('d_1') + 1
            d_n_idx = cols.get_loc(f'd_{NUM_TIME_STEPS}') + 1
            for row in tqdm(df.itertuples()):
                node_str = row[node_str_idx]
                a_ids = self.tree.get_ancestor_ids(node_str)
                ts = np.asarray(row[d_1_idx : d_n_idx+1])
                self.ts_data[a_ids] += ts
            
            features = pd.read_csv(feats_path, ',')
            feats_1 = np.asarray(
                features[['wday', 'month', 'snap_CA', 'snap_TX', 'snap_WI']]\
                    [:NUM_TIME_STEPS])
            feats_1 = minmax_scale(feats_1)

            feats_2 = features['event_type_1'][:NUM_TIME_STEPS]
            cats = list(feats_2.unique())
            cats.remove(np.nan)

            feats_2 = [[''] if isinstance(f, float) else [f] for f in feats_2]
            enc = OneHotEncoder(categories=[cats,], handle_unknown='ignore', sparse=False)
            feats_2 = enc.fit_transform(feats_2)

            self.feats = np.concatenate([feats_1, feats_2], axis=1)
            self.feats = self.feats.astype(np.float32)

            with open(pkl_path, 'wb') as fout:
                pickle.dump((self.tree, self.num_ts, self.ts_data, self.feats),
                            fout)


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
    
    def init_leaf_count(self):
        self.leaf_count = {}
        tot_leaves = self._count_rec(self.root)
        print(tot_leaves)
    
    def _count_rec(self, node_str):
        tot_count = 0
        for ch in self.children[node_str]:
            tot_count += self._count_rec(ch)
        if tot_count == 0:  # No children
            tot_count += 1
        self.leaf_count[node_str] = tot_count
        return tot_count


def main(_):
    tree = Tree()
    tree.insert_seq('food_2_1')
    tree.insert_seq('food_2_2')
    tree.insert_seq('hobbies_1_100')

    print(tree.parent)
    print(tree.children)
    print(tree.node_id)
    print(tree.id_node)

    data = M5Data()
    print(data.ts_data.dtype)
    print(data.feats.dtype)
    print(data.ts_data.shape)


if __name__ == "__main__":
    app.run(main)

