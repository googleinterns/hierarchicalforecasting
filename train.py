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
        flags.dataset, flags.model, flags.expt_name)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer,
        model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, expt_dir, max_to_keep=5,
        keep_checkpoint_every_n_hours=2)
    
    ckpt_path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_path)
    if ckpt_path is None:
        print('*' * 10, 'Checkpoint not found. Initializing ...')
    else:
        print('*' * 10, f'Checkpoint found. Restoring from {ckpt_path}')
    
    optimizer.learning_rate.assign(0.0001)
    @tf.function
    def train_step(feats, y_obs):
        with tf.GradientTape() as tape:
            mean, sig = model(feats, y_obs)
            loss = models.gaussian_nll(mean, sig, y_obs)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mean, sig

    metrics = Metrics(
        ['loss', 'abs_err'],
        expt_dir
    )

    for ep in range(int(ckpt.step), flags.train_epochs):
        print(f'Epoch {ep}')

        iterator = tqdm(data.tf_dataset(train=True))
        for i, (feats, y_obs) in enumerate(iterator):
            loss, mean, sig = train_step(feats, y_obs)
            metrics.update(
                {
                    'loss': loss,
                    'abs_err': tf.reduce_mean(tf.abs(mean - y_obs[1:]))
                }
            )
            if i % 100 == 0:
                mean_loss = metrics.metric_dict['loss'].result().numpy()
                iterator.set_description(f'Loss {mean_loss:.4f}')
        ckpt.step.assign_add(1)
        ckpt_manager.save()
        metrics.write(step=ep)
        metrics.write_others(
            {
                'learning_rate': optimizer.learning_rate
            },
            step=ep
        )


class Metrics:
    def __init__(self, metric_list, log_dir):
        self.metric_dict = {metric: keras.metrics.Mean() for metric in metric_list}
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def update(self, update_dict):
        for metric in update_dict:

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
    
    def write_others(self, dict, step):
        with self.writer.as_default():
            for metric in dict:
                tf.summary.scalar(metric, dict[metric], step=step)


# def update_mean(curr_mean, val, old_count):
#     return curr_mean + (val - curr_mean) / (old_count + 1)


if __name__ == "__main__":
    app.run(main)
