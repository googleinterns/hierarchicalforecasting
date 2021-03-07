from absl import flags

flags.DEFINE_string('expt', None, 'The name of the experiment dir')
flags.DEFINE_string('dataset', 'm5', 'The name of the experiment dir')
flags.DEFINE_string('reg_type', None, 'Type of hierarchy information to use')

flags.DEFINE_integer('train_epochs', 25, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', None, 'Batch size for the randomly sampled batch')

flags.DEFINE_integer('hist_len', 28, 'Length of the history provided as input')
flags.DEFINE_integer('train_pred', 28, 'Length of pred len during training')
flags.DEFINE_integer('test_pred', 7, 'Length of pred len during test/val')
flags.DEFINE_integer('val_windows', 5, 'Number of validation windows')
flags.DEFINE_integer('test_windows', 5, 'Number of validation windows')

flags.DEFINE_integer('fixed_lstm_hidden', 16,
                    'Number of LSTM hidden units in the local model')
flags.DEFINE_integer('node_emb_dim', 16,
                    'Dimension of the node embeddings')
flags.DEFINE_integer('random_seed', None,
                    'The random seed to be used for TF and numpy')
flags.DEFINE_integer("patience", 5, "Patience for early stopping")

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate')

FLAGS = flags.FLAGS
