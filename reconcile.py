import numpy as np
import pickle
import tensorflow as tf
# import tensorflow_addons as tfa

from tabulate import tabulate
from tqdm import tqdm
import pandas as pd


EPS = 1e-7


def erm_reconcile(y, yhat, leaf_mat, reg_mat, lamda=0.1, max_iters=10000, tol=1e-7):
    """Function to calculate reconciliation matrix through ERM.

    Args
    ----
    y: np.array
      actual values with shape n*t
    yhat: np.array
      base forecasts for all levels with shape n*t
      note that the last m ts are bottom level or leaves
    stree: np.array
      tree adjacency matrix n*m
      note that the last m*m block is I_m
    lambda: float
      regularization constant L1 loss
    max_iters: int
      maximum iters for gd
    tol: float
      tolerance for stopping applied on difference of losses

    Returns
    -------
    p_matrix: np.array
      reconciliation matrix of size m*n
    """
    n, m = leaf_mat.shape
    
    y_t = tf.convert_to_tensor(y, dtype=tf.float32)
    yhat_t = tf.convert_to_tensor(yhat, dtype=tf.float32)
    
    stree_t = tf.convert_to_tensor(leaf_mat.T, dtype=tf.float32)
    p_bu_t = tf.convert_to_tensor(reg_mat, dtype=tf.float32)

    p_matrix_t = tf.Variable(reg_mat, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    prev_loss = 0.0

    iterator = tqdm(range(max_iters), mininterval=1)
    for step in iterator:
        with tf.GradientTape() as tape:
            y_prime = tf.matmul(tf.matmul(yhat_t, p_matrix_t), stree_t)
            main_loss = tf.reduce_mean(tf.square(y_t - y_prime))
            reg_loss = tf.reduce_mean(tf.abs(p_matrix_t - p_bu_t))
            loss = main_loss + lamda * reg_loss
        grads = tape.gradient(loss, [p_matrix_t])
        optimizer.apply_gradients(zip(grads, [p_matrix_t]))
        iterator.set_description(f'Loss = {loss.numpy()}')
        if tf.abs(prev_loss - loss) < tol:
            print("Breaking at iter: {}".format(step))
            break
        prev_loss = loss
    return p_matrix_t.numpy()


def eval(all_y_pred, all_y_true, tree):
    level_dict = tree.levels
    results_list = []

    '''Compute metrics for all time series together'''
    results_dict = {}
    results_dict['level'] = 'all'
    for metric in METRICS:
        eval_fn = METRICS[metric]
        results_dict[metric] = eval_fn(all_y_pred, all_y_true)
    results_list.append(results_dict)

    '''Compute metrics for individual levels and their mean across levels'''
    mean_dict = {metric: [] for metric in METRICS}

    for d in level_dict:
        results_dict = {}
        sub_pred = all_y_pred[:, level_dict[d]]
        sub_true = all_y_true[:, level_dict[d]]
        for metric in METRICS:
            eval_fn = METRICS[metric]
            eval_val = eval_fn(sub_pred, sub_true)
            results_dict[metric] = eval_val
            mean_dict[metric].append(eval_val)
        results_dict['level'] = d
        results_list.append(results_dict)
    
    '''Compute the mean result of all the levels'''
    for metric in mean_dict:
        mean_dict[metric] = np.mean(mean_dict[metric])
    mean_dict['level'] = 'mean'
    results_list.append(mean_dict)
    
    df = pd.DataFrame(data=results_list)
    df.set_index('level', inplace=True)
    print(tabulate(df, headers='keys', tablefmt='psql'))


def mape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true).flatten()
    abs_val = np.abs(y_true).flatten()
    idx = np.where(abs_val > 0.1)
    mape = np.mean(abs_diff[idx]/abs_val[idx])
    return mape

def wape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_val = np.abs(y_true)
    wape = np.sum(abs_diff)/(np.sum(abs_val) + EPS)
    return wape

def smape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_mean = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(abs_diff/(abs_mean + EPS))
    return smape

METRICS = {'mape': mape, 'wape': wape, 'smape': smape}

def main():
    with open('notebooks/evals/val.pkl', 'rb') as fin:
        y_pred, y_true = pickle.load(fin)
    
    with open('notebooks/evals/test.pkl', 'rb') as fin:
        test_pred, test_true = pickle.load(fin)

    with open('data/favorita/data.pkl', 'rb') as fin:
        tree, ts_data, _ = pickle.load(fin)

    num_val = int(y_pred.shape[0] * 0.3)
    
    train_pred, train_true = y_pred[:-num_val], y_true[:-num_val]
    val_pred, val_true = y_pred[num_val:], y_true[num_val:]
    
    leaf_mat = tree.leaf_matrix
    num_leaf = np.sum(leaf_mat, axis=1, keepdims=True)
    leaf_mat /= num_leaf

    leaf_vec = tree.leaf_vector.astype(np.bool)
    sub_mat = leaf_mat[:, leaf_vec]
    reg_mat = sub_mat.copy()
    reg_mat[np.logical_not(leaf_vec)] = 0

    p_matrix = erm_reconcile(train_true, train_pred, sub_mat, reg_mat, lamda=1e4, max_iters=500)
    # p_matrix = reg_mat
    print(p_matrix)
    print(reg_mat)

    print('### Train')
    train_pred = (train_pred @ p_matrix) @ sub_mat.T
    eval(train_pred, train_true, tree)

    print('### Val')
    val_pred = (val_pred @ p_matrix) @ sub_mat.T
    eval(val_pred, val_true, tree)

    print('### Test')
    test_pred = (test_pred @ p_matrix) @ sub_mat.T
    eval(test_pred, test_true, tree)

if __name__=='__main__':
    main()
