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
    if flags.dataset == 'm5':
        data = data_loader.M5Data()
    else:
        raise ValueError(f'Unknown dataset {flags.dataset}')

    '''Create model'''
    if flags.model == 'fixed':
        model = models.FixedRNN(
            num_ts=data.num_ts, cat_dims=data.global_cat_dims,
            tree=data.tree)
    elif flags.model == 'random':
        model = models.RandomRNN(
            num_ts=data.num_ts, train_weights=train_weights, cat_dims=data.global_cat_dims)
    else:
        raise ValueError(f'Unknown model {flags.model}')
    
    '''Compute path to experiment directory'''
    model_name = flags.model
    if flags.reg_type is not None:
        model_name += '_' + flags.reg_type
    expt_dir = os.path.join('./logs',
        flags.dataset, model_name, flags.expt)

    step = tf.Variable(0)

    '''LR scheduling'''
    num_changes = 9
    boundaries = flags.train_epochs * np.linspace(0.1, 0.9, num_changes-1)
    boundaries = boundaries.astype(np.int32).tolist()

    lr = flags.learning_rate * np.asarray([0.5**(i+1) for i in range(num_changes)])
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

    eval_df, test_loss = model.eval(data)
    # sys.exit()
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

        iterator = tqdm(data.tf_dataset(train=True), mininterval=2)
        for i, (feats, y_obs, nid) in enumerate(iterator):
            reg_loss, loss = model.train_step(feats, y_obs, nid, optimizer)
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
        eval_df, test_loss = model.eval(data)

        tracked_loss = eval_df.loc['mean']['wape']
        if tracked_loss < best_loss:
            best_loss = tracked_loss
            best_check_path = ckpt_manager.latest_checkpoint
            pat = 0

            eval_save_path = os.path.join(expt_dir, "eval.pkl")
            with open(eval_save_path, "wb") as fout:
                pickle.dump(eval_df, fout)

            print("saved best result so far...")
        else:
            pat += 1
            if pat > flags.patience:
                print("early stopped with best loss: {}".format(best_loss))
                print("best model at: {}".format(best_check_path))
                break

        # summary.update(eval_dict)
        summary.write(step=step.numpy())

    '''Save embeddings to file'''
    # emb = model.get_node_emb(np.arange(data.num_ts))
    # emb = emb.numpy()
    # h = flags.hierarchy
    # if h is None:
    #     h = ""
    # fname = f'scratch/emb_{h}.pkl'
    # with open(fname, 'wb') as fout:
    #     pickle.dump(emb, fout)


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
