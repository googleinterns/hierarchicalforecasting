import tensorflow as tf
import numpy as np
import global_flags
import data_loader
import model

from tensorflow import keras
from absl import app

flags = global_flags.FLAGS

def main(_):
    data = data_loader.M5Data()
    dfrnn = model.DFRNN(num_ts=data.num_ts)

    for feats, y_obs in data.tf_dataset(train=True):
        print(feats.shape, y_obs.shape)
        mean, sig = dfrnn(feats, y_obs)
        loss = model.gaussian_nll(mean, sig, y_obs)
        print(mean.shape, sig.shape, loss)
        print(dfrnn.trainable_weights)
        break


if __name__ == "__main__":
    app.run(main)
