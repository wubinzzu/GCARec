"""
Paper: Hierarchical Gating Networks for Sequential Recommendation
Author: Chen Ma, Peng Kang, and Xue Liu
Reference: https://github.com/allenjack/HGN
@author: Zhongchuan Sun
"""

from model.AbstractRecommender import SeqAbstractRecommender
import tensorflow as tf
from util import log_loss, timer
import numpy as np
from util import l2_loss
from util.data_iterator import DataIterator
from util import pad_sequences
from util import batch_randint_choice


class HGN(SeqAbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(HGN, self).__init__(dataset, config)
        self.dataset = dataset
        self.lr = config.lr
        self.reg = config.reg
        self.seq_L = config.L
        self.seq_T = config.T
        self.embedding_size = config.embedding_size
        self.neg_samples = config.neg_samples
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        self.user_pos_train = self.dataset.get_user_train_dict(by_time=True)
        self.users_num, self.items_num = dataset.train_matrix.shape
        self.pad_id = self.items_num

        self.sess = sess

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seqs_ph = tf.placeholder(tf.int32, [None, self.seq_L], name="item_seqs")
        self.pos_item_ph = tf.placeholder(tf.int32, [None, self.seq_T], name="pos_item")
        self.neg_item_ph = tf.placeholder(tf.int32, [None, self.neg_samples], name="neg_item")
        self.predict_item = tf.concat([self.pos_item_ph, self.neg_item_ph], axis=1, name="item_to_predict")

    def _create_variable(self):
        self._reg_loss = 0
        stddev = 1.0 / self.embedding_size
        # init = tf.random.normal([self.users_num, self.embedding_size], mean=0.0, stddev=stddev)
        init = tf.random_normal([self.users_num, self.embedding_size], mean=0.0, stddev=stddev)
        self.user_embeddings = tf.Variable(init, dtype=tf.float32, name="user_embeddings")

        # init = tf.random.normal([self.items_num, self.embedding_size], mean=0.0, stddev=stddev)
        init = tf.random_normal([self.items_num, self.embedding_size], mean=0.0, stddev=stddev)
        item_embeddings = tf.Variable(init, dtype=tf.float32, name="item_embeddings")
        zero_pad = tf.zeros([1, self.embedding_size], name="padding1")
        self.item_embeddings = tf.concat([item_embeddings, zero_pad], axis=0)

        self.feature_gate_item = tf.layers.Dense(self.embedding_size, name="feature_gate_item")
        self.feature_gate_user = tf.layers.Dense(self.embedding_size, name="feature_gate_user")

        init = tf.initializers.he_uniform()
        self.instance_gate_item = tf.Variable(init([self.embedding_size, 1]),
                                              dtype=tf.float32, name="instance_gate_item")
        self.instance_gate_user = tf.Variable(init([self.embedding_size, self.seq_L]),
                                              dtype=tf.float32, name="instance_gate_user")

        init = tf.random.normal([self.items_num, self.embedding_size], mean=0.0, stddev=stddev)  # truncated_normal
        init = tf.random.normal([self.items_num, self.embedding_size], mean=0.0, stddev=stddev)  # truncated_normal
        W2 = tf.Variable(init, dtype=tf.float32, name="W2")
        zero_pad = tf.zeros([1, self.embedding_size], name="padding2")
        self.W2 = tf.concat([W2, zero_pad], axis=0)

        b2 = tf.Variable(tf.zeros([self.items_num]), dtype=tf.float32, name="b2")
        zero_pad = tf.zeros([1], name="padding3")
        self.b2 = tf.concat([b2, zero_pad], axis=0)

    def _forward(self, item_embs, user_emb):  # (b,l,d), (b,d)
        gate = tf.sigmoid(self.feature_gate_item(item_embs) +
                          tf.expand_dims(self.feature_gate_user(user_emb), axis=1))  # (b,l,d)

        self._reg_loss += l2_loss(*self.feature_gate_item.trainable_weights)
        self._reg_loss += l2_loss(*self.feature_gate_user.trainable_weights)

        # feature gating
        gated_item = tf.multiply(item_embs, gate)  # (b,l,d)

        # instance gating
        term1 = tf.matmul(gated_item, tf.expand_dims(self.instance_gate_item, axis=0))  # (b,l,d)x(1,d,1)->(b,l,1)
        term2 = tf.matmul(user_emb, self.instance_gate_user)  # (b,d)x(d,l)->(b,l)
        self._reg_loss += l2_loss(self.instance_gate_user, self.instance_gate_item)

        instance_score = tf.sigmoid(tf.squeeze(term1) + term2)  # (b,l)

        union_out = tf.multiply(gated_item, tf.expand_dims(instance_score, axis=2))  # (b,l,d)
        union_out = tf.reduce_sum(union_out, axis=1)  # (b,d)
        instance_score = tf.reduce_sum(instance_score, axis=1, keep_dims=True)
        union_out = union_out / instance_score  # (b,d)
        return union_out  # (b,d)

    def _train_rating(self, item_embs, user_emb, union_out):
        w2 = tf.nn.embedding_lookup(self.W2, self.predict_item)  # (b,2t,d)
        b2 = tf.gather(self.b2, self.predict_item)  # (b,2t)
        self._reg_loss += l2_loss(w2, b2)

        # MF
        term3 = tf.squeeze(tf.matmul(w2, tf.expand_dims(user_emb, axis=2)))  # (b,2t,d)x(b,d,1)->(b,2t,1)->(b,2t)
        res = b2 + term3  # (b,2t)

        # union-level
        term4 = tf.matmul(tf.expand_dims(union_out, axis=1), w2, transpose_b=True)  # (b,1,d)x(b,d,2l)->(b,1,2l)
        res += tf.squeeze(term4)  # (b,2t)

        # item-item product
        rel_score = tf.matmul(item_embs, w2, transpose_b=True)  # (b,l,d)x(b,d,2t)->(b,l,2t)
        rel_score = tf.reduce_sum(rel_score, axis=1)  # (b,2t)

        res += rel_score  # (b,2t)
        return res

    def _test_rating(self, item_embs, user_emb, union_out):
        # for testing
        w2 = self.W2  # (n,d)
        b2 = self.b2  # (n,)

        # MF
        res = tf.matmul(user_emb, w2, transpose_b=True) + b2  # (b,d)x(d,n)->(b,n)

        # union-level
        res += tf.matmul(union_out, w2, transpose_b=True)  # (b,d)x(d,n)->(b,n)

        # item-item product
        rel_score = tf.matmul(item_embs, w2, transpose_b=True)  # (b,l,d)x(d,n)->(b,l,n)
        rel_score = tf.reduce_sum(rel_score, axis=1)  # (b,n)

        res += rel_score
        return res  # (b,n)

    def build_graph(self):
        self._create_placeholder()
        self._create_variable()

        item_embs = tf.nn.embedding_lookup(self.item_embeddings, self.item_seqs_ph)  # (b,l,d)
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # (b,d)
        self._reg_loss += l2_loss(item_embs, user_emb)

        union_out = self._forward(item_embs, user_emb)  # (b,d)
        train_ratings = self._train_rating(item_embs, user_emb, union_out)  # (b,2t)

        pos_ratings, neg_ratings = tf.split(train_ratings, [self.seq_T, self.seq_T], axis=1)
        loss = tf.reduce_sum(log_loss(pos_ratings-neg_ratings))

        final_loss = loss + self.reg*self._reg_loss

        train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss)
        weights = self.feature_gate_item.trainable_weights + self.feature_gate_user.trainable_weights
        weights.extend([self.instance_gate_item, self.instance_gate_user])
        with tf.control_dependencies([train_opt]):
            self.train_opt = [tf.assign(weight, tf.clip_by_norm(weight, 1.0))
                              for weight in weights]

        # for testing
        self.bat_ratings = self._test_rating(item_embs, user_emb, union_out)  # (b,n)

    def train_model(self):
        users_list, item_seq_list, item_pos_list = self._generate_sequences()
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg in data:
                feed = {self.user_ph: bat_user,
                        self.item_seqs_ph: bat_item_seq,
                        self.pos_item_ph: bat_item_pos,
                        self.neg_item_ph: bat_item_neg,
                        }

                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _generate_sequences(self):
        self.user_test_seq = {}
        users_list, item_seq_list, item_pos_list = [], [], []
        seq_len = self.seq_L + self.seq_T
        uni_users = np.unique(list(self.user_pos_train.keys()))
        for user in uni_users:
            seq_items = self.user_pos_train[user]
            if len(seq_items) - seq_len >= 0:
                for i in range(len(seq_items), 0, -1):
                    if i-seq_len >= 0:
                        seq_i = seq_items[i-seq_len:i]
                        if user not in self.user_test_seq:
                            self.user_test_seq[user] = seq_i[-self.seq_L:]
                        users_list.append(user)
                        item_seq_list.append(seq_i[:self.seq_L])
                        item_pos_list.append(seq_i[-self.seq_T:])
                    else:
                        break
            else:
                seq_items = np.reshape(seq_items, newshape=[1, -1]).astype(np.int32)
                seq_items = pad_sequences(seq_items, value=self.pad_id, max_len=seq_len,
                                          padding='pre', truncating='pre')
                seq_i = np.reshape(seq_items, newshape=[-1])
                if user not in self.user_test_seq:
                    self.user_test_seq[user] = seq_i[-self.seq_L:]
                users_list.append(user)
                item_seq_list.append(seq_i[:self.seq_L])
                item_pos_list.append(seq_i[-self.seq_T:])
        return users_list, item_seq_list, item_pos_list

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c*self.neg_samples for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg

        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c, self.neg_samples])
            neg_items_list.extend(neg_items)
        return neg_items_list

    @timer
    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    # def predict(self, users, neg_items=None):
    #     bat_seq = [self.user_test_seq[u] for u in users]
    #     feed = {self.user_ph: users,
    #             self.item_seqs_ph: bat_seq
    #             }
    #     bat_ratings = self.sess.run(self.bat_ratings, feed_dict=feed)
    #     return bat_ratings
    def predict(self, users, items=None):
        users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_test_seq[u] for u in bat_user]
            feed = {self.user_ph: bat_user,
                    self.item_seqs_ph: bat_seq
                    }
            bat_ratings = self.sess.run(self.bat_ratings, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)

        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]

        return all_ratings
