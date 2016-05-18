###############################################################
#                        LBFGS THEANO
#                        No more fucking around
###############################################################

# THEANO
import numpy as np
from utils import castInt
# SCIPY
import random

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
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
