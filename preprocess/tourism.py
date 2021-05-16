import numpy as np
import pandas as pd
from tree import Tree
from sklearn.preprocessing import minmax_scale

import pickle


def main():
    data_path = 'data/tourism/TourismData_v3.csv'
    pkl_path = 'data/tourism/data.pkl'

    data = pd.read_csv(data_path, sep=',')

    raw_data = []
    for row in data.itertuples():
        raw_data.append(row[3:])
    raw_data = np.asarray(raw_data, dtype=np.float32)
    raw_data = raw_data.T

    raw_data = raw_data[:, :-36]

    _, t = raw_data.shape
    feats = np.arange(t)
    feats = np.vstack([feats, feats % 12]).T
    feats = minmax_scale(feats)

    print('feats', feats.shape)

    tree = Tree()

    nodelist = []
    item_nbr = 0
    for col in data.columns[2:]:
        col = col[:4]
        node_str = '_'.join(col)
        nodelist.append(node_str)
        tree.insert_seq(node_str)
    
    tree.precompute()

    ts_data = np.zeros((tree.num_nodes, t), dtype=np.float32)
    for i, node in enumerate(nodelist):
        node_id = tree.node_id[node]
        ts_data[node_id] = raw_data[i]
    
    ts_data = tree.leaf_matrix @ ts_data
    ts_data = ts_data.T
    print(f'ts_data: {ts_data.shape}')

    feats = (feats, [], [])
    with open(pkl_path, 'wb') as fout:
        pickle.dump((tree, ts_data, feats), fout)


if __name__ == '__main__':
    main()
