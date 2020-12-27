import os
import sys
import tensorflow as tf
import numpy as np
import pickle

import global_flags
import data_loader
import models

from tensorflow import keras
from absl import app
from tqdm import tqdm

flags = global_flags.FLAGS


def main(_):
    tf.random.set_seed(flags.random_seed)
    np.random.seed(flags.random_seed)

    print("FLAGS:")
    for flag in flags.flags_by_module_dict()["global_flags"]:
        print(f"\t--{flag.name}={flag._value}")

    # Load data
    if flags.dataset == "fav":
        data = data_loader.Favorita()
    else:
        raise ValueError(f"Unknown dataset {flags.dataset}")

    # Create model
    if flags.model == "fixed":
        model = models.FixedRNN(
            num_ts=data.num_ts, cat_dims=data.global_cat_dims, tree=data.tree
        )
    else:
        raise ValueError(f"Unknown model {flags.model}")

    ts_data = np.array([], dtype=np.float32).reshape((0, data.num_ts))
    iterator = tqdm(data.tf_dataset(train=None), mininterval=2)
    for feats, y_obs, nid in iterator:
        if ts_data.shape[0] == 0:
            print(ts_data.shape, y_obs.shape)
            y_obs = np.random.rand(flags.cont_len, data.num_ts).astype(
                np.float32
            )
            ts_data = np.concatenate([ts_data, y_obs])
        else:
            y_obs = ts_data[-flags.cont_len :]
        pred = model(feats, y_obs, nid)
        ts_data = np.concatenate([ts_data, pred])

    print(ts_data.shape, data.ts_data.shape)
    # mi = np.min(ts_data[flags.pred_hor:])
    # ma = np.max(ts_data[flags.pred_hor:])
    # ts_data = (ts_data - mi) / (ma - mi)
    with open("data/favorita/alternate_ts.pkl", "wb") as fout:
        pickle.dump(ts_data, fout)

    emb = model.get_node_emb(np.arange(data.num_ts))
    emb = emb.numpy()
    with open("scratch/gen_emb.pkl", "wb") as fout:
        pickle.dump(emb, fout)


if __name__ == "__main__":
    app.run(main)
