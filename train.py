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
    
    print('FLAGS:')
    for flag in flags.flags_by_module_dict()['global_flags']:
        print(f'\t--{flag.name}={flag._value}')

    '''Load data'''
    data = data_loader.Data()
    
    '''Create model'''
    model = models.FixedRNN(
        num_ts=data.num_ts, cat_dims=data.global_cat_dims,
        tree=data.tree)

    '''Compute path to experiment directory'''
    expt_dir = os.path.join('./logs',
        flags.dataset, flags.expt)

    step = tf.Variable(0)

    '''LR scheduling'''
    boundaries = flags.train_epochs * np.linspace(0.0, 1.0, flags.num_changes)
    boundaries = boundaries.astype(np.int32).tolist()

    lr = flags.learning_rate * np.asarray([0.5**i for i in range(flags.num_changes + 1)])
    lr = lr.tolist()

    sch = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=lr)
    optimizer = keras.optimizers.Adam(learning_rate=lr[0])
    # optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
    #                                  nesterov=True)

    '''Checkpointing'''
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

    model.eval(data, 'val')
    model.eval(data, 'test')
    
    # print(eval_df.loc['mean']['wape'])
    # summary.update(eval_dict)
    # summary.write(step=step.numpy())

    best_loss = 1e7
    pat = 0
    best_check_path = None

    while step.numpy() < flags.train_epochs:
        ep = step.numpy()
        print(f"Epoch {ep}")
        sys.stdout.flush()
        optimizer.learning_rate.assign(sch(step))

        iterator = tqdm(data.tf_dataset(mode='train'), mininterval=2)
        for i, (feats, y_obs, z, nid) in enumerate(iterator):
            reg_loss, loss = model.train_step(feats, y_obs, z, nid, optimizer)
            '''Train metrics'''
            summary.update({"train/reg_loss": reg_loss, "train/loss": loss})
            if i % 100 == 0:
                mean_loss = summary.metric_dict["train/reg_loss"].result().numpy()
                iterator.set_description(f"Reg + Loss {mean_loss:.4f}")
        step.assign_add(1)
        ckpt_manager.save()

        '''Other metrics'''
        summary.update({"train/learning_rate": optimizer.learning_rate.numpy()})

        '''Test metrics'''
        val_metrics, val_pred = model.eval(data, 'val')
        test_metrics, test_pred = model.eval(data, 'test')

        tracked_loss = val_metrics.loc['mean']['wape']
        if tracked_loss < best_loss:
            best_loss = tracked_loss
            best_check_path = ckpt_manager.latest_checkpoint
            pat = 0

            eval_save_path = os.path.join(expt_dir, "metrics.pkl")
            with open(eval_save_path, "wb") as fout:
                pickle.dump(
                    {'val': val_metrics, 'test': test_metrics}, fout)
            
            with open(os.path.join(expt_dir, 'val.pkl'), 'wb') as fout:
                pickle.dump(val_pred, fout)
            
            with open(os.path.join(expt_dir, 'test.pkl'), 'wb') as fout:
                pickle.dump(test_pred, fout)

            print("saved best result so far...")
        else:
            pat += 1
            if pat > flags.patience:
                print("Early stopping")
                break

        summary.write(step=step.numpy())

        '''Save embeddings to file'''
        emb = model.get_node_emb(np.arange(data.num_ts))
        emb = emb.numpy()
        fname = os.path.join(expt_dir, 'emb.pkl')
        with open(fname, 'wb') as fout:
            pickle.dump(emb, fout)

    print(f'Best model at {best_check_path} with loss = {best_loss}')


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
