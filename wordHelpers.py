###############################################################
#                        LBFGS THEANO
#                        No more fucking around
###############################################################

# THEANO
import numpy as np
from utils import castInt,castData
# SCIPY
import random

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']
vocabulary_size = len(vocab)
MAX_WORD_SIZE = 10
EOS = -1

def char2id(char):
  if char in vocab:
    return vocab.index(char)
  else:
    return 0
  
def id2char(dictid):
  if dictid != EOS:
    return vocab[dictid]
  else:
    return ' '

# Word helpers
def genRandWord():
  word_len = np.random.randint(1,MAX_WORD_SIZE)
  word = [id2char(np.random.randint(1,vocabulary_size)) for _ in range(word_len)]
  return ''.join(word)

def genRandBatch():
    word = genRandWord()
    batch = word2batch(word)
    rev_batch = word2batch(reverseWord(word))
    return castInt(batch),castInt(rev_batch)

# REVERSE HELPERS
def reverseWord(word):
  return word[::-1]

# BATCH CONVERSIONS
def batch2word(batch):
    return ''.join([id2char(i) for i in batch if i != EOS]) # Skip End of Sequence tag

def word2batch(word):
    batch = [char2id(letter) for letter in word] + [EOS] # Add End of Sequence tag
    return batch

def initFox():
    fox = []
    words = ['the','quick','brown','fox']
    for w in words:
        b = word2batch(w)
        r = word2batch(reverseWord(w))
        fox.append([b,castInt(r)])
    return fox

def id2onehot(i,vocab_size=vocabulary_size):
    oh = np.zeros((1,vocab_size))
    oh[0,i] = 1
    return oh

def words2text(words):
    text = ''
    for w in words:
        if len(w) > MAX_WORD_SIZE:
            words.remove(w)
        else:
            text += w + (' ' * (MAX_WORD_SIZE - len(w)))
    return text        

def load(hot,words=False):
    '''Load dat mnist'''

    filename = maybe_download('text8.zip', 31344016)

    text = read_data(filename)
    print('Data size %d' % len(text))


    valid_text = text[:VALID_SIZE]
    test_text = text[VALID_SIZE:VALID_SIZE+TEST_SIZE]
    # setting this for testing purposes
    test_text = 'the quick brown fox ' + test_text[20:]
    train_text = text[VALID_SIZE+TEST_SIZE:VALID_SIZE+TEST_SIZE+TRAIN_SIZE]

    # Convert each set into batch_size blocks of individual words
    if words:
        valid_words = valid_text.split(' ')
        test_words = test_text.split(' ')
        train_words = train_text.split(' ')

        print("number of train_words:",len(train_words))
        print("number of test_words:",len(test_words))
        print("number of valid_words:",len(valid_words))

        train_text = words2text(train_words)
        test_text = words2text(test_words)
        valid_text = words2text(valid_words)

    train_size = len(train_text)
    test_size = len(test_text)
    valid_size = len(valid_text)

    print(train_size, train_text[:64])
    print(test_size, test_text[:64])
    print(valid_size, valid_text[:64])

    trX = text2onehot(train_text,train_size)
    teX = text2onehot(test_text,test_size)
    tvX = text2onehot(valid_text,valid_size)
    trY = reverseData(trX,words)
    teY = reverseData(teX,words)
    tvY = reverseData(tvX,words)

    train_x = castData(trX)
    train_y = castData(trY)
    test_x = castData(teX)
    test_y = castData(teY)
    valid_x = castData(tvX)
    valid_y = castData(tvY)
    if not hot:
        train_y = T.cast(castData(trY), 'int32')
        test_y = T.cast(castData(teY), 'int32')
        valid_y = T.cast(castData(tvY), 'int32')
    return train_x,train_y,test_x,test_y,valid_x,valid_y
