import tensorflow as tf
import numpy as np
import global_flags
import sys

from tensorflow import keras
from tensorflow.keras import layers


flags = global_flags.FLAGS
MAX_FEAT_EMB_DIM = 50


class FixedRNN(keras.Model):
    def __init__(self, num_ts, cat_dims, tree):
        super().__init__()

        self.num_ts = num_ts
        self.cat_dims = cat_dims
        
        self.tree = tree
        # assert(flags.node_emb_dim == flags.fixed_lstm_hidden)
        self.node_emb = layers.Embedding(input_dim=self.num_ts,
            output_dim=flags.node_emb_dim, name='node_embed')
        self.input_scale = layers.Embedding(input_dim=self.num_ts,
            output_dim=1, embeddings_initializer=keras.initializers.ones,
            name='input_scale')
        
        self.embs = [
            layers.Embedding(input_dim=dim, output_dim=min(MAX_FEAT_EMB_DIM, (dim+1)//2))
            for dim in self.cat_dims
        ]

        self.encoder = layers.LSTM(flags.fixed_lstm_hidden,
                            return_state=True, time_major=True)
        self.decoder = layers.LSTM(flags.fixed_lstm_hidden,
                            return_sequences=True, time_major=True)
        self.output_trans = layers.Dense(flags.node_emb_dim, use_bias=False)
        self.output_scale = layers.Embedding(input_dim=self.num_ts,
            output_dim=1, embeddings_initializer=keras.initializers.ones,
            name='output_scale')
    
    def get_node_emb(self, nid):
        self.node_emb(np.asarray([0], dtype=np.int32))  # creates the emb matrix
        embs = self.node_emb.trainable_variables[0]
        if flags.hierarchy == 'additive':
            leaf_matrix = self.tree.leaf_matrix
            num_leaves = np.sum(leaf_matrix, axis=1, keepdims=True)
            leaf_matrix = leaf_matrix / num_leaves
            embs = tf.matmul(leaf_matrix, embs)
        elif flags.hierarchy == 'add_dev':
            print('\n\tHIERARCHY: Additive deviations')
            ancestor_matrix = self.tree.ancestor_matrix
            embs = tf.matmul(ancestor_matrix, embs)
        node_emb = tf.nn.embedding_lookup(embs, nid)
        return node_emb

    def assemble_feats(self, feats):
        feats_cont = feats[0]  # t x d
        feats_cat = feats[1]  # [t, t]
        feats_emb = [
            emb(feat) for emb, feat in zip(self.embs, feats_cat)  # t x e
        ]
        all_feats = feats_emb + [feats_cont]  # [t x *]
        all_feats = tf.concat(all_feats, axis=-1)  # t x d
        return all_feats
    
    def get_fixed(self, feats, y_prev, nid):
        input_scale = self.input_scale(nid)  # b x 1
        input_scale = tf.expand_dims(input_scale, 0)  # 1 x b x 1

        output_scale = self.output_scale(nid)  # b x 1
        output_scale = tf.expand_dims(output_scale, 0)  # 1 x b x 1

        y_prev = tf.expand_dims(y_prev, -1)  # t/2 x b x 1
        # y_prev = y_prev * input_scale  # t/2 x b x 1

        feats = tf.expand_dims(feats, 1)  # t x 1 x d
        feats = tf.repeat(feats, repeats=nid.shape[0], axis=1)  # t x b x d
        feats_prev = feats[:flags.pred_hor]  # t/2 x b x d
        feats_futr = feats[flags.pred_hor:]  # t/2 x b x d

        enc_inp = tf.concat([y_prev, feats_prev], axis=-1)  # t/2 x b x D'

        loadings = self.get_node_emb(nid)  # b x h
        loadings = tf.expand_dims(loadings, 0)  # 1 x b x h
        # loadings = loadings * output_scale

        _, h, c = self.encoder(inputs=enc_inp)  # b x h
        outputs = self.decoder(inputs=feats_futr, initial_state=(h, c))  # t x b x h
        outputs = self.output_trans(outputs)  # t x b x h

        local_effect = tf.reduce_sum(outputs * loadings, axis=-1)  # t x b
        return local_effect
    
    def regularizers(self):
        reg = 0.0
        if flags.hierarchy == 'laplacian':
            print('HIERARCHY: Graph Laplacian regularization')
            A = self.tree.adj_matrix
            D = np.sum(A, axis=0)
            D = np.diag(D)
            L = D - A  # n x n

            embs = self.node_emb.trainable_variables[0]  # n x e
            embs = tf.transpose(embs)  # e x n
            M = tf.matmul(embs, L)  # e x n
            dot = tf.reduce_sum(M * embs)
            reg = reg + flags.laplacian_weight * dot
        if flags.sparsity_weight > 0.0:
            print('\nSPARSITY: True')
            embs = self.node_emb.trainable_variables[0]  # n x e
            l1 = tf.reduce_mean(tf.square(embs))   ############ l2 norm
            reg = reg + flags.sparsity_weight * l1
        return reg

    @tf.function
    def call(self, feats, y_prev, nid):
        '''
        feats: t x d, t
        y_prev: t x b
        nid: b
        sw: b
        '''
        feats = self.assemble_feats(feats)  # t x d

        # Computing fixed effects
        fixed_effect = self.get_fixed(feats, y_prev, nid)  # t x b
        # final_output = tf.math.softplus(final_output)  # n
        return fixed_effect
    
    @tf.function
    def train_step(self, feats, y_obs, nid, optimizer):
        '''
        feats:  b x t x d, b x t
        y_obs:  b x t
        nid: b
        sw: b
        ''' 
        with tf.GradientTape() as tape:
            pred = self(feats, y_obs[:flags.pred_hor], nid)  # t x b
            # loss = tf.reduce_sum(tf.abs(pred - y_obs[-1, :]) * sw)
            rmse = tf.square(pred - y_obs[flags.pred_hor:])  # t x b
            rmse = tf.sqrt(tf.reduce_mean(rmse, axis=0))  # b
            rmse = tf.reduce_mean(rmse)
            loss = rmse + self.regularizers()

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        print(self.trainable_variables)

        return loss, rmse

    def eval(self, dataset, level_dict):
        pred_hor = flags.pred_hor

        for feats, y_obs, nid in dataset:
            assert(y_obs.numpy().shape[0] == 2 * pred_hor)
            assert(feats[0].numpy().shape[0] == 2 * pred_hor)

            y_pred = self(feats, y_obs[:pred_hor], nid)
            rmse = tf.square(y_pred - y_obs[pred_hor:])  # t x b
            rmse = tf.sqrt(tf.reduce_mean(rmse, axis=0))  # b
            rmse = rmse.numpy()

            return_dict = {}
            rmses = []
            for d in level_dict:
                sub_mean = np.mean(rmse[level_dict[d]])
                rmses.append(sub_mean)
                return_dict[f'test/rmse@{d}'] = sub_mean

            return_dict['test/rmse'] = np.mean(rmses)

        np.save('notebooks/evals.npy', y_pred)
        return return_dict