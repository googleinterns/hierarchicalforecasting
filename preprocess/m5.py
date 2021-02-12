import numpy as np
import pandas as pd
import os
import pickle

from absl import app
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, minmax_scale, StandardScaler
from tree import Tree

NUM_TIME_STEPS = 1941 - 28  # Offsetting by 28 # Starts from 1
START_IDX = 1

def main():
    data_path = ('data/m5/sales_train_evaluation.csv')
    feats_path = ('data/m5/calendar.csv')
    pkl_path = ('data/m5/data.pkl')

    print('Reading from csv ...')
    df = pd.read_csv(data_path, ',')
    tree = Tree()

    col_name = 'item_id' # 'dept_id'
    for item_id in df[col_name]:
        tree.insert_seq(item_id)
    tree.precompute()
    
    num_ts = tree.num_nodes
    ts_data = np.zeros((num_ts, NUM_TIME_STEPS), dtype=np.float32)

    cols = df.columns
    node_str_idx = cols.get_loc(col_name) + 1
    d_1_idx = cols.get_loc('d_1') + 1
    d_n_idx = cols.get_loc(f'd_{NUM_TIME_STEPS}') + 1
    for row in tqdm(df.itertuples()):
        node_str = row[node_str_idx]
        a_ids = tree.get_ancestor_ids(node_str)
        ts = np.asarray(row[d_1_idx : d_n_idx+1])
        ts_data[a_ids] += ts

    ts_data = ts_data.T
    
    features = pd.read_csv(feats_path, ',')
    feats = np.asarray(
        features[['wday', 'month', 'snap_CA', 'snap_TX', 'snap_WI']]\
            [:NUM_TIME_STEPS])
    feats = minmax_scale(feats)
    global_cont_feats = np.asarray(feats, dtype=np.float32)

    global_cat_feats = []
    global_cat_dims = []

    cat_feat_list = ['event_name_1', 'event_type_1']
    for cat_feat_name in cat_feat_list:
        feats = features[cat_feat_name][:NUM_TIME_STEPS].fillna('')
        feats = [[feat] for feat in feats]
        enc = OrdinalEncoder(dtype=np.int32)
        feats = enc.fit_transform(feats)
        global_cat_feats.append(np.asarray(feats, dtype=np.int32).ravel())
        global_cat_dims.append(len(enc.categories_[0]))

    feats = (global_cont_feats, global_cat_feats, global_cat_dims)
    with open(pkl_path, 'wb') as fout:
        pickle.dump((tree, ts_data, feats), fout)

if __name__ == '__main__':
    main()
