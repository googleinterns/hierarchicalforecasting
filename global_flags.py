from absl import flags

flags.DEFINE_string('expt', None, 'The name of the experiment dir')
flags.DEFINE_string('dataset', 'm5', 'The name of the experiment dir')
flags.DEFINE_string('model', 'fixed', 'The name of the experiment dir')
flags.DEFINE_string('m5dir', './data/m5', 'Path to the m5 data directory')
flags.DEFINE_string('hierarchy', None, 'Type of hierarchy information to use')

flags.DEFINE_integer('train_epochs', 25, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', None, 'Batch size for the randomly sampled batch')

# flags.DEFINE_integer('cont_len', 28, 'Length of the historical context')
flags.DEFINE_integer('pred_hor', 28, 'Length of the prediction horizon')

flags.DEFINE_integer('fixed_lstm_hidden', 16,
                    'Number of LSTM hidden units in the local model')
flags.DEFINE_integer('var_lstm_hidden', 10,
                    'Number of LSTM hidden units in the variance model')
flags.DEFINE_integer('node_emb_dim', 16,
                    'Dimension of the node embeddings')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate')
# flags.DEFINE_float('laplacian_weight', None,
#                    'Laplacian regularization weight')
flags.DEFINE_float('l2_reg_weight', 0.0,
                   'L2 regularization weight')
flags.DEFINE_float('l1_reg_weight', 0.0,
                   'L1 regularization weight')
flags.DEFINE_float('l2_weight_slack', 0.0,
                   'L2 regularization weight')

flags.DEFINE_boolean('overparam', False, 'Over parameterization')
flags.DEFINE_boolean('input_scaling', False,
                     'Learning input scale parameters')
flags.DEFINE_boolean('output_scaling', False,
                     'Learning output scale parameters')
flags.DEFINE_boolean('emb_as_inp', False,
                     'Whether to provide node embeddings as input')

FLAGS = flags.FLAGS
