from absl import flags

flags.DEFINE_string('expt', None, 'The name of the experiment dir')
flags.DEFINE_string('dataset', 'm5', 'The name of the experiment dir')
flags.DEFINE_string('model', 'fixed', 'The name of the experiment dir')
flags.DEFINE_string('m5dir', './data/m5', 'Path to the m5 data directory')
flags.DEFINE_string('hierarchy', None, 'Type of hierarchy information to use')

flags.DEFINE_integer('train_epochs', 35, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 500, 'Number of epochs to train')

flags.DEFINE_integer('cont_len', 28, 'Length of the historical context')
flags.DEFINE_integer('pred_hor', 28, 'Length of the prediction horizon')

flags.DEFINE_integer('global_lstm_hidden', 16,
                    'Number of LSTM hidden units in the global model')
flags.DEFINE_integer('local_lstm_hidden', 16,
                    'Number of LSTM hidden units in the local model')
flags.DEFINE_integer('var_lstm_hidden', 20,
                    'Number of LSTM hidden units in the variance model')
flags.DEFINE_integer('node_emb_dim', 20,
                    'Dimension of the node embeddings')

FLAGS = flags.FLAGS
