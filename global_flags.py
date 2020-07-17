from absl import flags

flags.DEFINE_string('expt', None, 'The name of the experiment dir')
flags.DEFINE_string('dataset', 'm5', 'The name of the experiment dir')
flags.DEFINE_string('model', 'simplernn', 'The name of the experiment dir')

flags.DEFINE_string('m5dir', './data/m5', 'Path to the m5 data directory')

flags.DEFINE_boolean('use_global_model', True, 'True if using global model')

flags.DEFINE_integer('train_epochs', 30, 'Number of epochs to train')

flags.DEFINE_integer('cont_len', 28, 'Length of the historical context')
flags.DEFINE_integer('pred_hor', 28, 'Length of the prediction horizon')

flags.DEFINE_integer('global_lstm_hidden', 10,
                    'Number of LSTM hidden units in the global model')
flags.DEFINE_integer('local_lstm_hidden', 10,
                    'Number of LSTM hidden units in the local model')
flags.DEFINE_integer('var_lstm_hidden', 20,
                    'Number of LSTM hidden units in the variance model')
flags.DEFINE_integer('feat_dim_local', 100,
                    'Number of LSTM hidden units in the local model')
flags.DEFINE_integer('feat_dim_global', 100,
                    'Number of LSTM hidden units in the local model')
flags.DEFINE_integer('node_emb_dim', 20,
                    'Dimension of the node embeddings')

FLAGS = flags.FLAGS
