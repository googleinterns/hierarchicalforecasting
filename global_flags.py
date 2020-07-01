from absl import flags

flags.DEFINE_string('m5dir', './data/m5', 'Path to the m5 data directory')

flags.DEFINE_integer('cont_len', 30, 'Length of the historical context')
flags.DEFINE_integer('pred_hor', 30, 'Length of the prediction horizon')

FLAGS = flags.FLAGS
