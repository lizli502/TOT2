import random
import numpy as np
from multiprocessing import Process, Queue


def random_neg(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, item2words, usernum, itemnum, vocab_size, batch_size, maxlen,  
                    threshold_user, threshold_item,result_queue, 
                    SEED, text_maxlen):
    def sample():
        user = np.random.randint(1, usernum + 1)
        user_ = user # without SSE
        while len(user_train[user]) <= 1: 
            user = np.random.randint(1, usernum + 1)
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        seq_words = []
        pos_words = []
        neg_words = []
        nxt = user_train[user][-1]
        idx = maxlen - 1
        
        # no SSE
        seq_ = np.zeros([maxlen], dtype=np.int32)

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            # seq[idx] = i
            # SSE for user side (2 lines)
            seq_[idx] = i
            if random.random() > threshold_item:    
                i = np.random.randint(1, itemnum + 1)
                nxt = np.random.randint(1, itemnum + 1)
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neg(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        
        # SSE for item side (2 lines)
        if random.random() > threshold_user:
            user = np.random.randint(1, usernum + 1)
        # equivalent to hard parameter sharing
        
        [item2words_seq, item2words_pos] = item2words
        for i in seq_:
            user_item = (user_, i)
            if i == 0 or user_item not in item2words_seq:
                seq_words_i = np.zeros(text_maxlen, dtype=int)
                pos_words_i = np.zeros(text_maxlen, dtype=int)
                neg_words_i = np.zeros(text_maxlen, dtype=int)
            else:
                # [0, 5, 3, 4, 6]
                ts = set(np.concatenate((item2words_seq[user_item], item2words_pos[user_item]), axis=0))
                seq_words_i = item2words_seq[user_item]            # [0, 5, 3, 4] 
                pos_words_i = item2words_pos[user_item]            # [0, 3, 4, 6]
                neg_words_i = []
                for w in pos_words_i:
                    if w == 0:
                        neg_words_i.append(0)
                    else:
                        neg_words_i.append(random_neg(1, vocab_size, ts))
                neg_words_i = np.asarray(neg_words_i, dtype=int)      # [0, 9, 7, 2]
            seq_words.append(seq_words_i)
            pos_words.append(pos_words_i)
            neg_words.append(neg_words_i)
        assert(len(seq_words) == maxlen)

        
        
        # shape: (maxlen, text_maxlen)
        seq_words = np.vstack(seq_words)
        pos_words = np.vstack(pos_words)
        neg_words = np.vstack(neg_words)
        #print('###############################################')
        #print('seq_words:\n {0}\npos_words:\n {1}\nneg_words\n {2}\n'.format(seq_words, pos_words, neg_words))
        #print('###############################################')

        return (user, seq, pos, neg, seq_words, pos_words, neg_words)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, item2words, usernum, itemnum, vocab_size, batch_size=64, maxlen=10,
                 threshold_user=1.0, threshold_item=1.0, n_workers=1, text_maxlen=64):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      item2words,
                                                      usernum,
                                                      itemnum,
                                                      vocab_size,
                                                      batch_size,
                                                      maxlen,
                                                      threshold_user,
                                                      threshold_item,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      text_maxlen
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
