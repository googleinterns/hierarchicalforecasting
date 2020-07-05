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

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

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
    

    @tf.function
    def train_step(feats, y_obs):
        with tf.GradientTape() as tape:
            mean, sig = model(feats, y_obs)
            loss = models.gaussian_nll(mean, sig, y_obs)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for ep in range(int(ckpt.step), flags.train_epochs):
        print(f'Epoch {ep}')
        loss_mean = 0.0
        iterator = tqdm(data.tf_dataset(train=True))
        model.reset_states()
        for i, (feats, y_obs) in enumerate(iterator):
            loss = train_step(feats, y_obs)
            loss_mean = update_mean(loss_mean, loss, i)
            if i % 100 == 0:
                iterator.set_description(f'Loss {loss_mean:.4f}')
        ckpt.step.assign_add(1)
        ckpt_manager.save()

            # break


def update_mean(curr_mean, val, old_count):
    return curr_mean + (val - curr_mean) / (old_count + 1)


if __name__ == "__main__":
    app.run(main)
