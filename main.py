import os
import time
import pickle
import argparse
import tensorflow as tf
from sampler1 import WarpSampler
from model_v1 import Model
#from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--user_hidden_units', default=50, type=int)
parser.add_argument('--item_hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_blocks_nlp', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--num_heads_nlp', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--dropout_rate_nlp', default=0.5, type=float)
parser.add_argument('--threshold_user', default=1.0, type=float)
parser.add_argument('--threshold_item', default=1.0, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--print_freq', default=100, type=int)
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--text_maxlen', default=64, type=int) # maxlen of a item's sequence 
parser.add_argument('--glove_emb_dim', default=100, type=int)
parser.add_argument('--loss_coef_nlp', default=1.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    params = '\n'.join([str(k) + ',' + str(v)
        for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
    print(params)
    f.write(params)

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset

# movielens1m 
#[item2words, vocab_size, word_index] = build_item_dict(args.dataset, args.text_maxlen)

# imdb reviews
#[item2words, vocab_size, word_index] = build_item_dict1('yelp_reviews8000.txt','yelp8000.txt', args.text_maxlen)
#[item2words, vocab_size, word_index] = build_item_dict1('imdb_reviews.txt','imdb.txt', args.text_maxlen)
[item2words, vocab_size, word_index] = build_item_dict1('amazon_book_reviews.txt','amazon_book.txt', args.text_maxlen)

glove_dir = 'data/glove'
glove_emb_matrix = build_emb_matrix(word_index, glove_dir, args.glove_emb_dim)

num_batch = len(user_train) // args.batch_size
cc = 0.0
max_len = 0
for u in user_train:
    cc += len(user_train[u])
    max_len = max(max_len, len(user_train[u]))
print("\nThere are {0} users {1} items \n".format(usernum, itemnum))
print("Average sequence length: {0}\n".format(cc / len(user_train)))
print("Maximum length of sequence: {0}\n".format(max_len))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, item2words, usernum, itemnum, vocab_size,
            batch_size=args.batch_size, maxlen=args.maxlen,
            threshold_user=args.threshold_user,
            threshold_item=args.threshold_item,
            n_workers=3, text_maxlen=args.text_maxlen)
model = Model(usernum, itemnum, vocab_size, glove_emb_matrix, args)
sess.run(tf.global_variables_initializer())

T = 0.0
t_test = evaluate(model, dataset, args, sess)
t_valid = evaluate_valid(model, dataset, args, sess)
print("[0, 0.0, {0}, {1}, {2}, {3}],".format(t_valid[0], t_valid[1], t_test[0], t_test[1]))

t0 = time.time()

for epoch in range(1, args.num_epochs + 1):
    for step in range(num_batch):
        u, seq, pos, neg, seq_words, pos_words, neg_words = sampler.next_batch()
        #print('###############################################')
        #print('seq_words:\n')
        #print(len(seq_words),len(seq_words[0]), len(seq_words[0][0]))
        #print('pos_words:\n')
        #print(len(pos_words))
        #print('neg_words:\n')
        #print(len(neg_words))
        #print('###############################################')

        auc, loss, loss_nlp, _ = sess.run([model.auc, model.loss, model.loss_nlp, model.train_op],
                                {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                 model.seq_words: seq_words, model.pos_words: pos_words,
                                 model.neg_words: neg_words, model.is_training: True})
        #if epoch % args.print_freq == 0:
        #    with open("attention_map_{}.pickle".format(step), 'wb') as fw:
        #        pickle.dump(attention, fw)
        #    with open("batch_{}.pickle".format(step), 'wb') as fw:
        #        pickle.dump([u, seq], fw)
        #    with open("user_emb.pickle", 'wb') as fw:
        #        pickle.dump(user_emb_table, fw)
        #    with open("item_emb.pickle", 'wb') as fw:
        #        pickle.dump(item_emb_table, fw)
    #print(loss, loss_nlp)
    if epoch % args.print_freq == 0:
        t1 = time.time() - t0
        T += t1
        #print 'Evaluating',
        t_test = evaluate(model, dataset, args, sess)
        t_valid = evaluate_valid(model, dataset, args, sess)
        #print ''
        #print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
        #epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
        print("[{0}, {1}, {2}, {3}, {4}, {5}],".format(epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
        #f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        #f.flush()
        t0 = time.time()

f.close()
sampler.close()
print("Done")
