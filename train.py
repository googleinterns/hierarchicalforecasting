import os
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
    # Load data
    if flags.dataset == 'm5':
        data = data_loader.M5Data()
    else:
        raise ValueError(f'Unknown dataset {flags.dataset}')

    # Create model
    if flags.model == 'fixed':
        model = models.FixedRNN(
            num_ts=data.num_ts, cat_dims=data.global_cat_dims,
            tree=data.tree)
    elif flags.model == 'random':
        model = models.RandomRNN(
            num_ts=data.num_ts, train_weights=train_weights, cat_dims=data.global_cat_dims)
    else:
        raise ValueError(f'Unknown model {flags.model}')
    
    # Compute path to experiment directory
    model_name = flags.model
    if flags.hierarchy is not None:
        model_name += '_' + flags.hierarchy
    expt_dir = os.path.join('./logs',
        flags.dataset, model_name, flags.expt)

    step = tf.Variable(0)

    # LR scheduling
    boundaries = flags.train_epochs * np.asarray([0.4, 0.7])
    boundaries = boundaries.astype(np.int32).tolist()

    lr = flags.learning_rate * np.asarray([1, 0.1, 0.01])
    lr = lr.tolist()

    sch = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=lr)
    optimizer = keras.optimizers.Adam()

    # Checkpointing
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer,
        model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, expt_dir, max_to_keep=5)
    ckpt_path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_path)

    if ckpt_path is None:
        print('*' * 10, 'Checkpoint not found. Initializing ...')
    else:
        print('*' * 10, f'Checkpoint found. Restoring from {ckpt_path}')
    
    summary = Summary(expt_dir)

    # eval_dict = model.eval(data.tf_dataset(train=False), data.tree.levels)
    # print(eval_dict)
    # summary.update(eval_dict)
    # summary.write(step=step.numpy())

    while step.numpy() < flags.train_epochs:
        ep = step.numpy()
        print(f'Epoch {ep}')
        optimizer.learning_rate.assign(sch(step))

        iterator = tqdm(data.tf_dataset(train=True), mininterval=2)
        for i, (feats, y_obs, nid, sw) in enumerate(iterator):
            loss = model.train_step(feats, y_obs, nid, sw, optimizer)
            # Train metrics
            summary.update({
                'train/loss': loss,
            })
            if i % 100 == 0:
                mean_loss = summary.metric_dict['train/loss'].result().numpy()
                iterator.set_description(f'Loss {mean_loss:.4f}')
        step.assign_add(1)
        ckpt_manager.save()

        # Other metrics
        summary.update({
            'train/learning_rate': optimizer.learning_rate.numpy()
        })
        
        # Test metrics
        eval_dict = model.eval(data.tf_dataset(train=False), data.tree.levels)
        print(eval_dict)
        summary.update(eval_dict)
        summary.write(step=step.numpy())

        eval_save_path = os.path.join(expt_dir, 'eval.pkl')
        with open(eval_save_path, 'wb') as fout:
            pickle.dump(eval_dict, fout)


class Summary:
    def __init__(self, log_dir):
        self.metric_dict = {}
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def update(self, update_dict):
        for metric in update_dict:
            if metric not in self.metric_dict:
                self.metric_dict[metric] = keras.metrics.Mean()
            self.metric_dict[metric].update_state(values=[update_dict[metric]])
    
    def write(self, step):
        with self.writer.as_default():
            for metric in self.metric_dict:
                tf.summary.scalar(metric, self.metric_dict[metric].result(), step=step)
        self.metric_dict = {}
        self.writer.flush()


if __name__ == "__main__":
    app.run(main)
