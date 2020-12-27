import tensorflow as tf
import numpy as np
import global_flags
import sys

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS
MAX_FEAT_EMB_DIM = 50


class FixedRNN(keras.Model):
    def __init__(self, num_ts, tree):
        super().__init__()

        self.num_ts = num_ts
        self.tree = tree
        # assert(flags.node_emb_dim == flags.fixed_lstm_hidden)
        self.node_emb = layers.Embedding(
            input_dim=self.num_ts,
            output_dim=flags.node_emb_dim,
            name="node_embed",
            embeddings_initializer=keras.initializers.RandomUniform(
                seed=flags.emb_seed
            ),
        )
        self.var_params = tf.Variable(
            np.random.uniform(size=[1, self.num_ts])
        )  # variance parameters for the dirichilet distributions in the cascade
        self.encoders = []
        self.decoders = []
        self.output_layers = []

        for i in range(flags.node_emb_dim):
            encoder = layers.LSTM(
                flags.fixed_lstm_hidden, return_state=True, time_major=True
            )
            decoder = layers.LSTM(
                flags.fixed_lstm_hidden, return_sequences=True, time_major=True
            )
            self.encoders.append(encoder)
            self.decoders.append(decoder)
            output_layer = layers.Dense(1, use_bias=False)
            self.output_layers.append(output_layer)
        # if flags.output_scaling:
        #     self.output_scale = layers.Embedding(input_dim=self.num_ts,
        #         output_dim=1, embeddings_initializer=keras.initializers.ones,
        #         name='output_scale')

    def get_normalized_emb(self):
        self.node_emb(np.asarray([0], dtype=np.int32))  # creates the emb matrix
        embs = tf.abs(self.node_emb.trainable_variables[0])
        embs = embs / tf.reduce_sum(embs, axis=1, keepdims=True)
        return embs

    def get_node_emb(self, nid):
        embs = self.get_normalized_emb()
        # if flags.hierarchy == 'additive':
        #     leaf_matrix = self.tree.leaf_matrix
        #     num_leaves = np.sum(leaf_matrix, axis=1, keepdims=True)
        #     leaf_matrix = leaf_matrix / num_leaves
        #     embs = tf.matmul(leaf_matrix, embs)
        # elif flags.hierarchy == 'add_dev':
        #     print('\n\tHIERARCHY: Additive deviations')
        #     ancestor_matrix = self.tree.ancestor_matrix
        #     embs = tf.matmul(ancestor_matrix, embs)
        # elif flags.hierarchy == 'sibling_reg':
        #     print('\n\tHIERARCHY: Sibling regularization')
        #     self.slack_emb(np.asarray([0], dtype=np.int32))  # creates the emb matrix
        #     slack_embs = self.slack_emb.trainable_variables[0]
        #     ancestor_matrix = self.tree.ancestor_matrix
        #     diag = np.diag(np.diag(ancestor_matrix))
        #     embs = tf.matmul(ancestor_matrix, embs) + \
        #            tf.matmul(ancestor_matrix - diag, slack_embs)
        node_emb = tf.nn.embedding_lookup(embs, nid)
        return node_emb

    # def assemble_feats(self, feats):
    #     feats_cont = feats[0]  # t x d
    #     feats_cat = feats[1]  # [t, t]
    #     feats_emb = [
    #         emb(feat) for emb, feat in zip(self.embs, feats_cat)  # t x e
    #     ]
    #     all_feats = feats_emb + [feats_cont]  # [t x *]
    #     all_feats = tf.concat(all_feats, axis=-1)  # t x d
    #     return all_feats

    def get_fixed(self, feats, y_prev, nid):
        y_prev = tf.expand_dims(y_prev, -1)  # t/2 x b x 1

        node_emb = self.get_node_emb(nid)  # b x h
        feats = tf.expand_dims(feats, 1)  # t x 1 x d
        feats = tf.repeat(feats, repeats=nid.shape[0], axis=1)  # t x b x d

        feats_prev = feats[: flags.cont_len]  # t/2 x b x d
        feats_futr = feats[flags.cont_len :]  # t/2 x b x d

        enc_inp = tf.concat([y_prev, feats_prev], axis=-1)  # t/2 x b x D'

        loadings = tf.expand_dims(node_emb, 0)  # 1 x b x h

        outputs = []
        for e, d, o in zip(self.encoders, self.decoders, self.output_layers):
            _, h, c = e(inputs=enc_inp)  # b x h
            output = d(inputs=feats_futr, initial_state=(h, c))  # t x b x h
            output = o(output)  # t x b x 1
            outputs.append(output)

        outputs = tf.concat(outputs, axis=-1)  # t x b x h

        # if flags.emb_as_inp or flags.overparam:
        #     outputs = self.output_trans(outputs)  # t x b x h

        fixed_effect = tf.reduce_sum(outputs * loadings, axis=-1)  # t x b
        # fixed_effect = tf.math.softplus(fixed_effect)

        return fixed_effect

    def regularizers(self, nid):
        # reg = 0.0
        # if flags.l2_reg_weight > 0.0:
        #     print('\nL2 reg: True')
        #     embs = self.node_emb.trainable_variables[0]  # n x e
        #     l2 = tf.reduce_mean(tf.square(embs))
        #     reg = reg + flags.l2_reg_weight * l2
        # if flags.l2_weight_slack > 0.0:
        #     print('\nL2 reg slack: True')
        #     slack_embs = self.slack_emb.trainable_variables[0]
        #     l2 = tf.reduce_mean(tf.square(slack_embs))
        #     reg = reg + flags.l2_weight_slack * l2
        # if flags.l1_reg_weight > 0.0:
        #     node_emb = self.get_node_emb(np.arange(self.num_ts))
        #     l1 = tf.reduce_mean(tf.abs(node_emb), axis=1)
        #     l1 = tf.reduce_mean(self.tree.leaf_vector * l1)
        #     reg = reg + flags.l1_reg_weight * l1

        A = self.tree.adj_matrix  # n x n
        A = np.expand_dims(A, axis=0)  # 1 x n x n

        embs = self.get_normalized_emb()  # n x e
        embs = tf.transpose(embs)  # e x n
        emb_1 = tf.expand_dims(embs, axis=-1)  # e x n x 1
        emb_2 = tf.expand_dims(embs, axis=-2)  # e x 1 x n

        edge_diff = emb_1 * A - emb_2 * A
        edge_diff = tf.abs(edge_diff)  ## Change here for L1/L2 regularization
        reg = flags.l2_reg_weight * tf.reduce_mean(edge_diff)

        return reg

    def dirichilet_cascade_mle(self):
        """Implement dirichilet cascade mle.
        """
        A = self.tree.adj_matrix  # n x n
        A = np.expand_dims(A, axis=0)  # 1 x n x n

        embs = self.get_normalized_emb()  # n x e
        embs = tf.transpose(embs)  # e x n
        emb_1 = tf.expand_dims(embs, axis=-1)  # e x n x 1
        var_params = tf.expand_dims(self.var_params, axis=-1)  # 1 x n x 1
        emb_1 = emb_1 * var_params  # e x n x 1
        emb_2 = tf.expand_dims(embs, axis=-2)  # e x 1 x n
        neg_mle = (emb_1 * A - 1.0) * tf.math.log(emb_2 * A)  # e x n x n
        neg_mle = tf.reduce_sum(neg_mle, axis=0)  # n x n
        b_alpha = emb_1 * A
        b_alpha_1 = tf.reduce_sum(tf.math.lgamma(b_alpha), axis=0)
        b_alpha_2 = tf.math.lgamma(tf.reduce_sum(b_alpha, axis=0))
        total_loss = b_alpha_1 - b_alpha_2 - neg_mle
        return flags.mle_reg_weight * tf.reduce_mean(total_loss)

    @tf.function
    def call(self, feats, y_prev, nid):
        """
        feats: t x d, t
        y_prev: t x b
        nid: b
        sw: b
        """
        # feats = self.assemble_feats(feats)  # t x d

        # Computing fixed effects
        fixed_effect = self.get_fixed(feats, y_prev, nid)  # t x b
        # final_output = tf.math.softplus(final_output)  # n
        return fixed_effect

    @tf.function
    def train_step(self, feats, y_obs, nid, optimizer):
        """
        feats:  b x t x d, b x t
        y_obs:  b x t
        nid: b
        sw: b
        """
        with tf.GradientTape() as tape:
            pred = self(feats, y_obs[: flags.cont_len], nid)  # t x 1
            mae = tf.abs(pred - y_obs[flags.cont_len :])  # t x 1
            mae = tf.reduce_mean(mae)
            loss = mae + self.regularizers(nid)

        tv = self.trainable_variables
        # tv = [v for v in self.trainable_variables if 'embed' in v.name]
        grads = tape.gradient(loss, tv)
        optimizer.apply_gradients(zip(grads, tv))
        return loss, mae

    def eval(self, dataset, level_dict):
        cont_len = flags.cont_len
        return_dict = {f"test/rmse@{d}": [] for d in level_dict}

        for feats, y_obs, nid in dataset:
            assert y_obs.numpy().shape[0] == cont_len + 1
            assert feats.numpy().shape[0] == cont_len + 1

            pred = self(feats, y_obs[: flags.cont_len], nid)  # t x 1
            mae = tf.abs(pred - y_obs[flags.cont_len :])  # t x 1
            mae = mae.numpy().ravel()

            for d in level_dict:
                sub_mean = np.mean(mae[level_dict[d]])
                return_dict[f"test/rmse@{d}"].append(sub_mean)

        rmses = []
        for key in return_dict:
            mean = np.mean(return_dict[key])
            rmses.append(mean)
            return_dict[key] = mean

        return_dict["test/rmse"] = np.mean(rmses)

        # np.save('notebooks/evals.npy', y_pred)
        return return_dict
