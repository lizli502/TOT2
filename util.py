import sys
import copy
import random
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]



# used for ml1m/ml10m dataset
def build_item_dict(fname, text_maxlen):
    item2words_seq = {}
    item2words_pos = {}
    items = pd.read_csv("data/{0}/movies_info.dat".format(fname), sep="::", 
            names=["id", "name", "genres", "plot"], engine='python')
    items["plot"] = items["plot"].str.lower()
    text = items["plot"]
    ids = items['id']
    
    cc = 0.0
    max_len = 0
    for t in text:
        cc += len(t)
        max_len = max(max_len, len(t))
    print("Average sequence length of text: {0}\n".format(cc / len(text)))
    print("Maximum length of sequence of text: {0}\n".format(max_len))
	
    max_features = 2000000
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(list(text))
    seq = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(seq, maxlen=text_maxlen + 1, padding='post')
    vocab_size = len(tokenizer.word_index)
    print('Unique tokens in text: {0}\n'.format(vocab_size))

    for i, s in zip(ids, seq):
        item2words_seq[i] = s[:-1]
        for t, x in zip(s, range(text_maxlen + 1)):
            if t != 0:
                break
            else:
                continue
        item2words_pos[i] = np.concatenate((s[:x],s[(x + 1):]), axis=0)
    item2words = [item2words_seq, item2words_pos]
    word_index = tokenizer.word_index
    return [item2words, vocab_size, word_index] 

# used for imdb dataset
# imdb_reviews.txt imdb.txt
def build_item_dict1(fname1, fname2, text_maxlen):
    item2words_seq = {}
    item2words_pos = {}
    text = []
    user_item = []
    
    with open("data/{0}".format(fname1)) as f1, open("data/{0}".format(fname2)) as f2:
        for line1, line2 in zip(f1, f2):
            tmp = line2.strip().split()
            user = int(tmp[0])
            item = int(tmp[1])
            user_item.append((user, item))
            review = line1.strip().lower()
            text.append(review)
        
        cc = 0.0
        max_len = 0
        for t in text:
            cc += len(t)
            max_len = max(max_len, len(t))
        print("Average sequence length of text: {0}\n".format(cc / len(text)))
        print("Maximum length of sequence of text: {0}\n".format(max_len))
            
        max_features = 2000000
        tokenizer = Tokenizer(num_words=max_features, filters='')
        tokenizer.fit_on_texts(text)
        seq = tokenizer.texts_to_sequences(text)
        
        seq_copy = seq
        # sliding window for reviews
        target_len = text_maxlen + 1
        for i in range(0, len(seq)):
            rs = seq[i] 
            rs_len = len(review)
            if rs_len > target_len:
                start = random.randint(0, rs_len - target_len + 1)
                end = start + target_len + 1
                new_rs = rs[start : end]
                print(start, end, new_rs)
                exit()
            else:
                new_rs = pad_sequences(rs, maxlen=target_len, padding='post', truncating='post')
            seq[i] = new_rs
        print(rs)
        print(new_rs)

        print(seq_copy[0])
        print(seq[0])
                
        print(seq_copy[1])
        print(seq[1])

        print(seq_copy[2])
        print(seq[2])
        exit()

        vocab_size = len(tokenizer.word_index)
            
        for i, s in zip(user_item, seq):
            item2words_seq[i] = s[:-1]
            for t, x in zip(s, range(text_maxlen + 1)):
                if t != 0:
                    break
            item2words_pos[i] = np.concatenate((s[:x],s[(x + 1):]), axis=0)
        item2words = [item2words_seq, item2words_pos]
        word_index = tokenizer.word_index
        #print(user_item[0])        
        #print(item2words_seq[(user_item[0])])
        #print(item2words_pos[(user_item[0])])
        

        return [item2words, vocab_size, word_index]


def build_emb_matrix(word_index, glove_dir, glove_emb_dim):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.{0}d.txt'.format(glove_emb_dim)))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # emb dim
    embedding_matrix = np.zeros((len(word_index) + 1, glove_emb_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-1, 1, (glove_emb_dim))
    return embedding_matrix



def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        #print(predictions)
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < args.k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 1000 == 0:
            #print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < args.k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            #print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
