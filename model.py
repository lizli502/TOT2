from modules import *


class Model():
    def __init__(self, usernum, itemnum, vocab_size, glove_emb, args, reuse=tf.AUTO_REUSE):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        # (B, T, L)
        self.seq_words = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.text_maxlen))
        self.pos_words = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.text_maxlen))
        self.neg_words = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.text_maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        mask_nlp = tf.expand_dims(tf.reshape(tf.to_float(tf.not_equal(self.seq_words, 0)), 
                                             [-1, args.text_maxlen]), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # glove word embedding table
            glove_emb_table = tf.get_variable(name="glove_embedding_table",
                                              shape=(1 + vocab_size, 100),
                                              initializer=tf.constant_initializer(glove_emb),
                                              #trainable=False)
                                              trainable=True)
            # shape = (B, T, L, 100)
            glove_emb_table = tf.concat((tf.zeros(shape=[1, args.glove_emb_dim]), glove_emb_table[1:]), 0)
            seq_words_emb = tf.nn.embedding_lookup(glove_emb_table, self.seq_words)
            pos_words_emb = tf.nn.embedding_lookup(glove_emb_table, self.pos_words)
            neg_words_emb = tf.nn.embedding_lookup(glove_emb_table, self.neg_words)

            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.item_hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.item_emb_table = item_emb_table
            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.item_hidden_units + args.user_hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            #self.seq += t

            # User Encoding
            u0_latent, user_emb_table = embedding(self.u[0],
                                                 vocab_size=usernum + 1,
                                                 num_units=args.user_hidden_units,
                                                 zero_pad=False,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="user_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.user_emb_table = user_emb_table
            # Has dim: B by C
            u_latent = embedding(self.u,
                                 vocab_size=usernum + 1,
                                 num_units=args.user_hidden_units,
                                 zero_pad=False,
                                 scale=True,
                                 l2_reg=args.l2_emb,
                                 scope="user_embeddings",
                                 with_t=False,
                                 reuse=reuse
                                 )
            # Change dim to B by T by C
            self.u_latent = tf.tile(tf.expand_dims(u_latent, 1), [1, tf.shape(self.input_seq)[1], 1])

            # Concat item embedding with user embedding
            self.hidden_units = args.item_hidden_units + args.user_hidden_units
            self.seq = tf.reshape(tf.concat([self.seq, self.u_latent], 2),
                                  [tf.shape(self.input_seq)[0], -1, self.hidden_units])
            self.seq += t
            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks
            self.attention = []
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq, attention = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")
                    self.attention.append(attention)
                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)
        
        user_emb = tf.reshape(self.u_latent, [tf.shape(self.input_seq)[0] * args.maxlen, 
                                              args.user_hidden_units])

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)

        pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units])
        neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])

        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, self.hidden_units])
        
        
        
        
        
        
        # reshape seq_emb and seq_words_emb to form new input_seq
        # (B * T, L, 100)
        seq_words_emb = tf.reshape(seq_words_emb, [tf.shape(self.input_seq)[0] * args.maxlen, 
                                                   args.text_maxlen, 100]) 
        seq_emb_nlp =  tf.tile(tf.expand_dims(seq_emb, 1), [1, args.text_maxlen, 1])
        # form input: (B * T, L, 200) if 50 for user, 50 for item
        self.hidden_units_nlp = self.hidden_units + 100
        seq_nlp = tf.reshape(tf.concat([seq_words_emb, seq_emb_nlp], 2),
                                  [tf.shape(self.input_seq)[0] * args.maxlen, args.text_maxlen, self.hidden_units_nlp])
        # add positional encoding and dropout
        t_nlp, pos_emb_table_nlp = embedding(
                tf.tile(tf.expand_dims(tf.range(args.text_maxlen), 0), 
                        [tf.shape(self.input_seq)[0] * args.maxlen, 1]),
                vocab_size=args.text_maxlen,
                num_units=self.hidden_units_nlp,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_nlp",
                reuse=reuse,
                with_t=True
                )
        seq_nlp += t_nlp
        # Dropout
        seq_nlp = tf.layers.dropout(seq_nlp, 
                                    rate=args.dropout_rate_nlp,
                                    training=tf.convert_to_tensor(self.is_training)
                                    )
        seq_nlp *= mask_nlp
	# Build blocks
        self.attention_nlp = []
        for i in range(args.num_blocks_nlp):
            with tf.variable_scope("num_blocks_%d_nlp" % i):
                # Self-attention
                seq_nlp, attention_nlp = multihead_attention(queries=normalize(seq_nlp),
                                                   keys=seq_nlp,
                                                   num_units=self.hidden_units_nlp,
                                                   num_heads=args.num_heads_nlp,
                                                   dropout_rate=args.dropout_rate_nlp,
                                                   is_training=self.is_training,
                                                   causality=False, # not sure
                                                   #causality=True,
                                                   scope="self_attention_nlp")
                self.attention_nlp.append(attention_nlp)
                 # Feed forward
                seq_nlp = feedforward(normalize(seq_nlp), 
                                           num_units=[self.hidden_units_nlp, self.hidden_units_nlp],
                                           dropout_rate=args.dropout_rate_nlp, 
                                           is_training=self.is_training,
                                           )
                seq_nlp *= mask_nlp

            seq_nlp = normalize(seq_nlp)        
        seq_nlp = tf.reshape(seq_nlp, [tf.shape(self.input_seq)[0] * args.maxlen * args.text_maxlen, self.hidden_units_nlp])
        
        pos_words_emb = tf.reshape(pos_words_emb, [tf.shape(self.input_seq)[0] * args.maxlen * args.text_maxlen, 100])
        seq_emb_nlp = tf.reshape(seq_emb_nlp, [-1, self.hidden_units])
        pos_words_emb = tf.reshape(tf.concat([pos_words_emb, seq_emb_nlp], 1), 
                                   [-1, self.hidden_units_nlp])
        neg_words_emb = tf.reshape(neg_words_emb, [tf.shape(self.input_seq)[0] * args.maxlen * args.text_maxlen, 100]) 
        neg_words_emb = tf.reshape(tf.concat([neg_words_emb, seq_emb_nlp], 1), 
                                   [-1, self.hidden_units_nlp])

        self.pos_logits_nlp = tf.reduce_sum(pos_words_emb * seq_nlp, -1)
        self.neg_logits_nlp = tf.reduce_sum(neg_words_emb * seq_nlp, -1)
        istarget_nlp = tf.reshape(tf.to_float(tf.not_equal(self.pos_words, 0)), [tf.shape(self.input_seq)[0] * args.maxlen * args.text_maxlen])
        self.loss_nlp = tf.reduce_sum(
                            - tf.log(tf.sigmoid(self.pos_logits_nlp) + 1e-24) * istarget_nlp -
                            tf.log(1 - tf.sigmoid(self.neg_logits_nlp) + 1e-24) * istarget_nlp
                            ) / tf.reduce_sum(istarget_nlp)
        
        

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        
        test_user_emb = tf.tile(tf.expand_dims(u0_latent, 0), [101, 1])
        # combine item and user emb
        test_item_emb = tf.reshape(tf.concat([test_item_emb, test_user_emb], 1), [-1, self.hidden_units])

        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        
        
        #self.loss = tf.reduce_sum(
        #    - tf.log(tf.exp(tf.sigmoid(self.pos_logits)) + 1e-24) * istarget +
        #    tf.log(tf.exp(tf.sigmoid(self.pos_logits)) + tf.exp(tf.sigmoid(self.neg_logits)) + 1e-24) * istarget 
        #) / tf.reduce_sum(istarget)

        #self.loss = tf.reduce_sum(-tf.log(1 + tf.exp(tf.sigmoid(self.pos_logits) - tf.sigmoid(self.neg_logits))) * istarget) / tf.reduce_sum(istarget)
            
        #self.loss = tf.reduce_sum(-tf.maximum(0.0, self.pos_logits - self.neg_logits - 0.001) * istarget) / tf.reduce_sum(istarget)

        #self.loss = tf.reduce_sum(-tf.log(tf.clip_by_value(tf.sigmoid(self.pos_logits - self.neg_logits), 1e-5, 1)) * istarget) / tf.reduce_sum(istarget)
        #self.loss = tf.reduce_sum(-tf.square(tf.maximum(self.pos_logits - self.neg_logits - 100, 0)) * istarget) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)
        
        self.loss += self.loss_nlp * args.loss_coef_nlp

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        tf.summary.scalar('auc', self.auc)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
