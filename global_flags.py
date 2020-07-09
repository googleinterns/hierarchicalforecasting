from absl import flags

flags.DEFINE_string('expt', None, 'The name of the experiment dir')
flags.DEFINE_string('dataset', 'm5', 'The name of the experiment dir')
flags.DEFINE_string('model', 'dfrnn', 'The name of the experiment dir')

flags.DEFINE_string('m5dir', './data/m5', 'Path to the m5 data directory')

flags.DEFINE_integer('train_epochs', 25, 'Number of epochs to train')

flags.DEFINE_integer('cont_len', 30, 'Length of the historical context')
flags.DEFINE_integer('pred_hor', 30, 'Length of the prediction horizon')

flags.DEFINE_integer('prin_comp', 100,
                    'Number of principal components for the global model')
flags.DEFINE_integer('global_lstm_hidden', 100,
                    'Number of LSTM hidden units in the global model')
flags.DEFINE_integer('local_lstm_hidden', 10,
                    'Number of LSTM hidden units in the local model')

FLAGS = flags.FLAGS
