import numpy as np
import pandas as pd
from tree import Tree
from sklearn.preprocessing import minmax_scale

import pickle


def main():
    data_path = 'data/wiki/data.csv'
    cleaned_path = 'data/wiki/data_cleaned.csv'
    pkl_path = 'data/wiki/data.pkl'

    with open(data_path, 'r') as fin:
        data = fin.readlines()

    with open(cleaned_path, 'w') as fout:
        for line in data:
            line = line.split(',')
            if len(line) > 554:
                line = line[:3] + ["_".join(line[3:-550])] + line[-550:]
            assert(len(line) == 554)
            line = ",".join(line)
            fout.write(line)
    
    data = pd.read_csv(cleaned_path, sep=',')
    cols = ['Category(countrycode)', 'Subcategory(access)', 'Subsubcategory(agent)', 'Page']
    for c in cols:
        data[c] = data[c].astype('category').cat.codes

    raw_data = []
    for row in data.itertuples():
        raw_data.append(row[5:])
    raw_data = np.asarray(raw_data, dtype=np.float32)

    _, t = raw_data.shape
    feats = np.arange(t)
    feats = np.vstack([feats, feats % 7]).T
    feats = minmax_scale(feats)

    print('feats', feats.shape)

    tree = Tree()

    nodelist = []
    item_nbr = 0
    for row in data.itertuples():
        node_str = [str(it) for it in row[1:5]]
        node_str = '_'.join(node_str)
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
