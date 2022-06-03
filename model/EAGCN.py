import numpy as np
from util import timer
import tensorflow as tf
from model.AbstractRecommender import SocialAbstractRecommender
from util.tool import inner_product, l2_loss
import scipy.sparse as sp


class EAGCN(SocialAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(EAGCN, self).__init__(dataset, conf)
        self.embedding_size = int(conf["embedding_size"])
        self.r_alpha = float(conf["r_alpha"])
        self.num_epochs = int(conf["epochs"])
        self.reg = float(conf["reg"])
        self.reg_w = float(conf["reg_w"])
        self.lr = float(conf["learning_rate"])
        self.layer_size = int(conf["layer_size"])
        self.verbose = int(conf["verbose"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess = sess

    def _create_recsys_adj_mat(self):
        user_item_idx = [[u, i] for (u, i), r in self.dataset.train_matrix.todok().items()]
        user_list, item_list = list(zip(*user_item_idx))

        self.user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="user_idx")
        self.item_idx = tf.constant(item_list, dtype=tf.int32, shape=None, name="item_idx")

        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))  # (m+n)*(m+n)
        adj_mat = tmp_adj + tmp_adj.T

        return self._normalize_spmat(adj_mat)

    def _create_social_adj_mat(self):
        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        u1_idx, u2_idx = list(zip(*uu_idx))

        self.u1_idx = tf.constant(u1_idx, dtype=tf.int32, shape=None, name="u1_idx")
        self.u2_idx = tf.constant(u2_idx, dtype=tf.int32, shape=None, name="u2_idx")

        u1_idx = np.array(u1_idx, dtype=np.int32)
        u2_idx = np.array(u2_idx, dtype=np.int32)
        ratings = np.ones_like(u1_idx, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (u1_idx, u2_idx)),
                                shape=(self.num_users, self.num_users))
        adj_mat = tmp_adj + tmp_adj.T
        return self._normalize_spmat(adj_mat)

    def _normalize_spmat(self, adj_mat):
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        print('use the pre adjcency matrix')
        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")

    def _create_variables(self):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]), name='user_embeddings')
        self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]), name='item_embeddings')

        # uu_weight project user embeddings into social space
        self.uu_weight = tf.Variable(initializer([self.embedding_size, self.embedding_size]), name='uu_weight')
        # ui_weight project user embeddings into recommendation space
        self.ui_weight = tf.Variable(initializer([self.embedding_size, self.embedding_size]), name='ui_weight')

        self.weights = dict()
        Gate_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=tf.sqrt(
            tf.div(2.0, self.embedding_size + self.embedding_size)))
        self.weights.setdefault("interest_social",
                                tf.Variable(Gate_initializer([self.embedding_size, self.embedding_size]),
                                            dtype=tf.float32))
        self.weights.setdefault("interest_self",
                                tf.Variable(Gate_initializer([self.embedding_size, self.embedding_size]),
                                            dtype=tf.float32))

    def _gcn(self, norm_adj, init_embeddings):
        ego_embeddings = init_embeddings
        all_embeddings = [ego_embeddings]
        for k in range(self.layer_size):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        return all_embeddings

    def _social_gcn(self):
        norm_adj = self._create_social_adj_mat()
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        user_embeddings = tf.matmul(self.user_embeddings, self.uu_weight)

        user_embeddings = self._gcn(norm_adj, user_embeddings)
        return user_embeddings  # m*d

    def _recsys_gcn(self):
        norm_adj = self._create_recsys_adj_mat()
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        user_embeddings = tf.matmul(self.user_embeddings, self.ui_weight)

        ego_embeddings = tf.concat([user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = self._gcn(norm_adj, ego_embeddings)
        user_embeddings, item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return user_embeddings, item_embeddings

    def _fast_loss(self, embeddings_a, embeddings_b, index_a, index_b, alpha):
        term1 = tf.matmul(embeddings_a, embeddings_a, transpose_a=True)
        term2 = tf.matmul(embeddings_b, embeddings_b, transpose_a=True)
        loss1 = tf.reduce_sum(tf.multiply(term1, term2))

        embed_a = tf.nn.embedding_lookup(embeddings_a, index_a)
        embed_b = tf.nn.embedding_lookup(embeddings_b, index_b)
        pos_ratings = inner_product(embed_a, embed_b)

        loss2 = (alpha - 1) * tf.reduce_sum(tf.square(pos_ratings)) - 2.0 * alpha * tf.reduce_sum(pos_ratings)
        return loss1 + loss2

    def build_graph(self):
        # ---------- matrix factorization -------
        self._create_placeholder()
        self._create_variables()
        social_embeddings = self._social_gcn()
        user_embeddings, item_embeddings = self._recsys_gcn()

        gating = tf.sigmoid(tf.matmul(user_embeddings, self.weights["interest_self"]) +
                            tf.matmul(social_embeddings, self.weights["interest_social"]))  # b,d

        final_user_embedding = (1 - gating) * user_embeddings + gating * social_embeddings

        recsys_loss = self._fast_loss(final_user_embedding, item_embeddings, self.user_idx, self.item_idx, self.r_alpha)

        self.obj_loss = recsys_loss + self.reg * l2_loss(self.user_embeddings, self.item_embeddings) + \
                        self.reg_w * l2_loss(self.ui_weight, self.uu_weight, self.weights["interest_social"],
                                             self.weights["interest_self"])

        self.update_opt = tf.train.AdagradOptimizer(self.lr).minimize(self.obj_loss)

        # for the testing phase
        self.item_embeddings_final = tf.Variable(tf.zeros([self.num_items, self.embedding_size]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.num_users, self.embedding_size]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, final_user_embedding),
                           tf.assign(self.item_embeddings_final, item_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.user_ph)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.num_epochs):
            _, _ = self.sess.run([self.update_opt, self.obj_loss])
            if epoch >= 1:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_ph: user_ids})
        else:
            ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_ph: user_ids})
            ratings = [ratings[idx][test_items] for idx, test_items in enumerate(candidate_items)]

        return ratings
