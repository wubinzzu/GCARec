from model.AbstractRecommender import SocialAbstractRecommender
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from util import DataIterator, timer
from util.tool import csr_to_user_dict_bytime,csr_to_time_dict,csr_to_user_dict
from util.cython.random_choice import batch_randint_choice
from util import pad_sequences
import math
epsilon = 1e-9


def l2_distance(a, b, name="euclidean_distance"):
    return tf.norm(a - b, ord='euclidean', axis=-1, name=name)
class GAST_Final(SocialAbstractRecommender):
    def __init__(self,sess,dataset, conf):  
        super(GAST_Final, self).__init__(dataset, conf)
        self.learning_rate = float(conf["learning_rate"])
        self.embedding_size = int(conf["embedding_size"])
        self.num_epochs= int(conf["epochs"])
        self.reg_mf = float(conf["reg_mf"])
        self.reg_W = float(conf["reg_w"])
        self.beta = float(conf["beta"])
        self.n_layers = conf['n_layers']
        self.batch_size= int(conf["batch_size"])
        self.verbose= int(conf["verbose"])
        self.seq_L = conf["seq_l"]
        self.target_T = conf['target_l']
        self.neg_samples = conf['neg_samples']
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset = dataset
        self.SocialtrainDict = self._get_SocialDict()
        self.norm_adj = self.create_adj_mat()
        self.adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        train_matrix, train_time_matrix, test_matrix = dataset.train_matrix, dataset.time_matrix, dataset.test_matrix
        test_time_matrix = dataset.test_time_matrix
        self.user_pos_train = csr_to_user_dict_bytime(train_time_matrix, train_matrix)
        self.user_pos_time = csr_to_time_dict(train_time_matrix)
        self.user_pos_test = csr_to_user_dict(test_matrix)
        self.user_test_time = {}
        for user_id in range(self.num_users):
            seq_timeone = test_time_matrix[user_id, self.user_pos_test[user_id][0]]
            seq_times = self.user_pos_time[user_id]
            content_time = list()
            size = len(seq_times)
            for index in range(min([self.seq_L, size])):
                deltatime_now = abs(seq_times[size-index-1] - seq_timeone) / (3600 * 24)
                if deltatime_now <= 0.5:
                    deltatime_now = 0.5
                content_time.append(math.log(deltatime_now))
            if (size < self.seq_L):
                content_time = content_time + [self.num_items for _ in range(self.seq_L - len(content_time))]
            self.user_test_time[user_id] = content_time
        self.sess=sess

    def _get_SocialDict(self):
        #find items rated by trusted neighbors only
        SocialDict = {}
        for u in range(self.num_users):
            trustors = self.social_matrix[u].indices
            if len(trustors)>0:
                SocialDict[u] = trustors.tolist()
            else:
                SocialDict[u] = [self.num_users]
        return SocialDict

    @timer
    def create_adj_mat(self):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix
    def _create_gcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.embeddings["user_embeddings"], self.embeddings["item_embeddings"]], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None,self.target_T], name = "item_input_pos")
            self.social_user_input = tf.placeholder(tf.int32, [None,None], name = "social_user_input")
            self.item_input_recent = tf.placeholder(tf.int32, shape = [None,self.seq_L], name = "item_input_recent")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None,self.neg_samples], name = "item_input_neg")
            self.relative_position_input = tf.placeholder(tf.float32, [None, self.seq_L], name=".relative_position")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embeddings = dict()
            #embeding_initializer = tf.contrib.layers.xavier_initializer()
            embeding_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            user_embeddings = tf.Variable(embeding_initializer([self.num_users, self.embedding_size]), dtype=tf.float32)
            self.embeddings.setdefault("user_embeddings", user_embeddings)

            self.c1 = tf.Variable(embeding_initializer(shape=[self.num_users, self.embedding_size]), dtype=tf.float32) # (users, embedding_size)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.social_embeddings = tf.concat([self.c1, self.c2], 0, name='social_embeddings')

            item_embeddings = tf.Variable(embeding_initializer([self.num_items, self.embedding_size]), dtype=tf.float32)
            self.embeddings.setdefault("item_embeddings", item_embeddings)

            seq_embeddings = tf.Variable(embeding_initializer([self.num_items, self.embedding_size]),
                                              dtype=tf.float32)
            self.embeddings.setdefault("seq_item_embeddings", seq_embeddings)

            self.user_embeddings, self.target_item_embeddings = self._create_gcn_embed()

            self.d2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='d2')
            
            self.seq_item_embeddings = tf.concat([seq_embeddings,self.d2], 0)
            
            self.item_biases = tf.Variable(tf.truncated_normal(shape=[self.num_items], mean=0.0, stddev=0.01),
                name='item_biases', dtype=tf.float32)  #(items)
            
            self.W = tf.Variable(tf.truncated_normal(shape=[2*self.embedding_size, self.embedding_size],mean=0.0, stddev=0.01),
                                 name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size],mean=0.0, stddev=0.01),
                                 name='Bias_for_MLP', dtype=tf.float32, trainable=True)

            
            self.Ws = tf.Variable(tf.truncated_normal(shape=[2*self.embedding_size, self.embedding_size],mean=0.0, stddev=0.01),
                                 name='Weights_social', dtype=tf.float32, trainable=True)
            self.bs = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size],mean=0.0, stddev=0.01),
                                 name='Bias_social', dtype=tf.float32, trainable=True)

            self.weight_mlp =tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size],mean=0.0, stddev=0.01),
                                 name='Weights_mlp', dtype=tf.float32, trainable=True)
        
    def Item_attention(self, user_embeddings, seq_item_embeddings):
        with tf.name_scope("item_attention"):
            b = tf.shape(user_embeddings)[0]
            n = tf.shape(seq_item_embeddings)[1]
            user_embeddings = tf.tile(user_embeddings, tf.stack([1, self.seq_L, 1])) #b*K*d

            relative_times = tf.tile(tf.expand_dims(self.relative_position_input,-1),tf.stack([1,1,self.embedding_size]))# b,L,d
            weight_mlp = tf.tile(tf.expand_dims(self.weight_mlp,1), tf.stack([b,n,1]))
            relative_position_embeddings = tf.multiply(weight_mlp,relative_times)

            mask_mat = tf.to_float(tf.not_equal(self.relative_position_input, self.num_items))  # (b, l)
            mask_mat = tf.tile(tf.expand_dims(mask_mat,-1),tf.stack([1,1,self.embedding_size]))#(b,l,d)
            embedding_short = tf.concat([seq_item_embeddings,tf.nn.tanh(relative_position_embeddings)*mask_mat], 2)  #b*K*2d
            #embedding_short = seq_item_embeddings
            # (bL, 3d) * (3d, d) + (1, d)
            MLP_output = tf.matmul(tf.reshape(embedding_short, shape=[-1, 2*self.embedding_size]), self.W) + self.b #(bK,d)
            MLP_output = tf.nn.tanh(MLP_output)#(bL,d)
            MLP_output = tf.reshape(MLP_output, shape=[-1, self.seq_L, self.embedding_size])  # (b,k,d)
            A_ = tf.reduce_sum(tf.multiply(MLP_output, user_embeddings),-1) # (b, k)

            # softmax for not mask features
            exp_A_pos = tf.exp(A_)
            exp_sum_pos = tf.reduce_sum(exp_A_pos, 1, keepdims=True)  # (b, k)

            A_ = tf.expand_dims(tf.div(exp_A_pos, exp_sum_pos), 2)  # (b, k, 1)

            return tf.reduce_sum(A_* seq_item_embeddings, 1, keepdims=True)
    
    def Social_attention(self, user_embeddings, short_embeddings, social_user_embeding):
        with tf.name_scope("Social_attention"):
            n = tf.shape(social_user_embeding)[1]
            user_embeddings = tf.tile(user_embeddings, tf.stack([1, n, 1]))  # b*K*d
            short_embeddings = tf.tile(short_embeddings,tf.stack([1,n,1]))#(b,n,e)
            
            social_embeddings = tf.concat([short_embeddings, social_user_embeding], 2)  #b*K*3d
            
            MLP_output = tf.matmul(tf.reshape(social_embeddings,shape=[-1, 2*self.embedding_size]), self.Ws) + self.bs #(b*n,e)
            MLP_output = tf.nn.tanh(MLP_output)

            MLP_output = tf.reshape(MLP_output, shape=[-1, n, self.embedding_size])  # (b,n,e)
            A = tf.reduce_sum(tf.multiply(MLP_output, user_embeddings), -1)  # (b, n)

            exp_A = tf.exp(A)
            mask_mat = tf.to_float(tf.not_equal(self.social_user_input, self.num_users)) #(b, n)
            exp_A = mask_mat*exp_A
            exp_sum = tf.reduce_sum(exp_A, 1, keepdims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum+epsilon,tf.constant(self.beta, tf.float32, [1]))
            A_ = tf.expand_dims(tf.div(exp_A, exp_sum)*mask_mat,2) # (b, n, 1)

            return tf.reduce_sum(A_ * social_user_embeding,1, keepdims=True)
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            # embedding look up
            self.user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            user_embs = tf.expand_dims(self.user_embedding, axis=1)  # (b, 1, d)
            self.item_embedding_recent = tf.nn.embedding_lookup(self.seq_item_embeddings, self.item_input_recent) #b,L,d
            self.batch_size_b = tf.shape(self.user_embedding)[0]
            
            self.short_embeddings = self.Item_attention(user_embs,self.item_embedding_recent)
            
            self.social_user_embeding = tf.nn.embedding_lookup(self.social_embeddings,self.social_user_input) # b*S*d
            self.social_attention_embeddings = self.Social_attention(user_embs, self.short_embeddings, self.social_user_embeding)
            
            self.final_user_embeddings = user_embs + self.short_embeddings + self.social_attention_embeddings # # b,1,d


            self.item_embedding_pos = tf.nn.embedding_lookup(self.target_item_embeddings, self.item_input) #b,T,d
            self.item_embedding_neg = tf.nn.embedding_lookup(self.target_item_embeddings, self.item_input_neg) #b,N,d
            self.tar_item_bias = tf.gather(self.item_biases, tf.concat([self.item_input, self.item_input_neg], axis=1))

            tar_item_embs = tf.concat([self.item_embedding_pos, self.item_embedding_neg], axis=1)
            logits = tf.squeeze(tf.matmul(self.final_user_embeddings, tar_item_embs, transpose_b=True), axis=1) + self.tar_item_bias

            self.pos_logits, self.neg_logits = tf.split(logits, [self.target_T, self.neg_samples], axis=1)

            # predict_vector_pos = self.final_user_embeddings*self.item_embedding_pos
            # self.output = self.tar_item_bias_pos + tf.reduce_sum(predict_vector_pos, 1)
            #
            # predict_vector_neg = self.final_user_embeddings*self.item_embedding_neg
            #
            # self.output_neg = self.tar_item_bias_neg + tf.reduce_sum(predict_vector_neg, 1)
                   
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_inference()
        pos_loss = tf.reduce_sum(-tf.log(tf.sigmoid(self.pos_logits) + 1e-24))
        neg_loss = tf.reduce_sum(-tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24))
        loss = pos_loss + neg_loss

        user_embeddings_reg = tf.nn.embedding_lookup(self.embeddings["user_embeddings"],self.user_input)
        pos_embeddings_reg = tf.nn.embedding_lookup(self.embeddings["item_embeddings"], self.item_input)
        neg_embeddings_reg = tf.nn.embedding_lookup(self.embeddings["item_embeddings"], self.item_input_neg)
        L_embeddings_reg = tf.nn.embedding_lookup(self.embeddings["seq_item_embeddings"], self.item_input_recent)
        social_embeddings_reg = tf.nn.embedding_lookup(self.social_embeddings, self.social_user_input)
        L2_emb = tf.reduce_sum(tf.square(user_embeddings_reg))+ tf.reduce_sum(tf.square(pos_embeddings_reg))+\
                      tf.reduce_sum(tf.square(neg_embeddings_reg))+tf.reduce_sum(tf.square(L_embeddings_reg))+\
                      tf.reduce_sum(tf.square(social_embeddings_reg))+tf.reduce_sum(tf.square(self.tar_item_bias))

        L2_weight =  tf.reduce_sum(tf.square(self.W))+ tf.reduce_sum(tf.square(self.b))+\
                          tf.reduce_sum(tf.square(self.Ws))+tf.reduce_sum(tf.square(self.bs))
        self.loss = loss+ self.reg_mf*L2_emb + self.reg_W*L2_weight

        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.e2 = tf.constant(0.0, tf.float32, [1], name='e2')
        self.all_logits = tf.matmul(tf.squeeze(self.final_user_embeddings, axis=1), self.target_item_embeddings, transpose_b=True) + self.item_biases

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c*self.neg_samples for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.num_items, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg

        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c,self.neg_samples])
            neg_items_list.extend(neg_items)
        return neg_items_list

    def _generate_sequences(self):
        self.user_test_seq = {}
        user_list, item_seq_list, item_pos_list, user_social_list, relative_time_list= [], [], [], [], []

        for user_id in range(self.num_users):
            seq_items = self.user_pos_train[user_id]
            social_friends = self.SocialtrainDict[user_id]
            seq_times = self.user_pos_time[user_id]
            for index_id in range(len(seq_items)):
                if index_id < self.target_T: continue
                content_data = list()
                content_time = list()
                self.seq_timeone = seq_times[min([index_id + 1, len(seq_items) - 1])]
                for cindex in range(max([0, index_id - self.seq_L-self.target_T+1]), index_id-self.target_T+1):
                    content_data.append(seq_items[cindex])
                    deltatime_now = abs((seq_times[cindex] - self.seq_timeone)) / (3600 * 24)
                    if deltatime_now <= 0.5:
                        deltatime_now = 0.5
                    content_time.append(math.log(deltatime_now))
                if (len(content_data) < self.seq_L):
                    content_data = content_data + [self.num_items for _ in range(self.seq_L - len(content_data))]
                    content_time = content_time + [self.num_items for _ in range(self.seq_L - len(content_time))]

                user_list.append(user_id)
                item_seq_list.append(content_data)
                relative_time_list.append(content_time)
                item_pos_list.append(seq_items[index_id - self.target_T + 1:index_id + 1])
                user_social_list.append(social_friends)

            user_id_seq = seq_items[-min([len(seq_items), self.seq_L]):]
            if (len(seq_items) < self.seq_L):
                user_id_seq = user_id_seq + [self.num_items for _ in range(self.seq_L - len(user_id_seq))]
            self.user_test_seq[user_id] = user_id_seq

        return user_list, item_seq_list, item_pos_list, user_social_list, relative_time_list
    
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.user_pos_train = csr_to_user_dict_bytime(self.dataset.time_matrix, self.dataset.train_matrix)
        users_list, item_seq_list, item_pos_list, user_social_list, relative_time_list = self._generate_sequences()
        for epoch in  range(self.num_epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list, user_social_list,relative_time_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg, bat_user_social, bat_relative_time in data:
                bat_user_social = pad_sequences(bat_user_social, value=self.num_users)
                feed = {self.user_input: bat_user,
                        self.item_input_recent: bat_item_seq,
                        self.item_input: bat_item_pos,
                        self.item_input_neg: bat_item_neg,
                        self.social_user_input:bat_user_social,
                        self.relative_position_input:bat_relative_time}
                _,_=self.sess.run([self.opt, self.loss], feed_dict=feed)
            # print(epoch)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))
    def evaluate_model(self):
        return self.evaluator.evaluate(self)
    def predict(self, user_ids, candidate_items_userids=None):
        users = DataIterator(user_ids, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_test_seq[u] for u in bat_user]
            bat_social = [self.SocialtrainDict[u] for u in bat_user]
            bat_social = pad_sequences(bat_social, value=self.num_users)
            bat_times = [self.user_test_time[u] for u in bat_user]
            feed = {self.user_input: bat_user,
                    self.item_input_recent: bat_seq,
                    self.social_user_input: bat_social,
                    self.relative_position_input: bat_times}
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)

        if candidate_items_userids is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(candidate_items_userids)]

        return all_ratings