import os
import tensorflow as tf
import numpy as np
import global_flags
import data_loader
import models

from tensorflow import keras
from absl import app
from tqdm import tqdm

flags = global_flags.FLAGS

def main(_):

    if flags.dataset == 'm5':
        data = data_loader.M5Data()
    else:
        raise ValueError(f'Unknown dataset {flags.dataset}')

    if flags.model == 'dfrnn':
        model = models.DFRNN(num_ts=data.num_ts)
    else:
        raise ValueError(f'Unknown model {flags.model}')
    
    expt_dir = os.path.join('./logs',
        flags.dataset, flags.model, flags.expt)

    step = tf.Variable(0)
    sch = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[15, 20], values=[1e-4, 1e-5, 1e-6])
    optimizer = keras.optimizers.Adam()

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

    while step.numpy() < flags.train_epochs:
        ep = step.numpy()
        print(f'Epoch {ep}')
        optimizer.learning_rate.assign(sch(step))

        iterator = tqdm(data.tf_dataset(train=True))
        for i, (feats, y_obs) in enumerate(iterator):
            loss, abs_err = model.train_step(feats, y_obs, optimizer)
            # Train metrics
            summary.update({
                'train/loss': loss,
                'train/abs_err': abs_err
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
        eval_dict = model.eval(data.tf_dataset(train=False))
        summary.update(eval_dict)
        summary.write(step=ep)


class Summary:
    def __init__(self, log_dir):
        self.metric_dict = {}
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def update(self, update_dict):
        for metric in update_dict:
            if metric not in self.metric_dict:
                self.metric_dict[metric] = keras.metrics.Mean()
            self.metric_dict[metric].update_state(values=[update_dict[metric]])
    
    def reset(self):
        for metric in self.metric_dict:
            self.metric_dict[metric].reset_states()
    
    def write(self, step):
        with self.writer.as_default():
            for metric in self.metric_dict:
                tf.summary.scalar(metric, self.metric_dict[metric].result(), step=step)
                self.metric_dict[metric].reset_states()
        self.writer.flush()


# def update_mean(curr_mean, val, old_count):
#     return curr_mean + (val - curr_mean) / (old_count + 1)


if __name__ == "__main__":
    app.run(main)
