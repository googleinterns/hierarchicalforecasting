import numpy as np
import tensorflow.compat.v2 as tf


def erm_reconcile(y, yhat, stree, lamda=0.1, max_iters=100000, tol=1e-7):
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
    n, m = stree.shape
    t = y.shape[1]
    p_bu = np.hstack([np.zeros(shape=(m, n - m)), np.eye(m)])
    y_t = tf.convert_to_tensor(y, dtype=tf.float32)
    yhat_t = tf.convert_to_tensor(yhat, dtype=tf.float32)
    stree_t = tf.convert_to_tensor(stree, dtype=tf.float32)
    p_bu_t = tf.convert_to_tensor(p_bu, dtype=tf.float32)

    p_matrix_t = tf.Variable(tf.random.normal([m, n]), dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9)
    prev_loss = 0.0
    for step in range(max_iters):
        with tf.GradientTape() as tape:
            y_prime = tf.matmul(stree_t, tf.matmul(p_matrix_t, yhat_t))
            main_loss = tf.reduce_mean(tf.abs(y_t - y_prime))
            reg_loss = tf.reduce_mean(tf.abs(p_matrix_t - p_bu_t))
            loss = main_loss + lamda * reg_loss
        grads = tape.gradient(loss, [p_matrix_t])
        optimizer.apply_gradients(zip(grads, [p_matrix_t]))
        if tf.abs(prev_loss - loss) < tol:
            print("Breaking at iter: {}".format(step))
            break
        prev_loss = loss
    return p_matrix_t.numpy()
