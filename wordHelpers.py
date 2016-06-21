# THEANO
import numpy as np
from utils import castInt,castData

# SCIPY
import random

#vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']
vocab = [' ', '-', ',', '.', ';', ':', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x','\n']
vocab_size = len(vocab)
MAX_WORD_SIZE = 10
EOS = -1 # Last element of vocab is End of Sequence tag


class WordHelper:
  def __init__(self,vocab,max_word_size=10,custom_eos=None):
    if custom_eos is None:
      self.eos_char = u'\xa4' # Arbitrary end-of-sequence tag, I think it looks cool
    else:
      self.eos_char = custom_eos
    #vocab += [self.eos_char]
    self.eos = -1
    self.vocab = vocab
    self.vocab_size = len(vocab)
    self.max_word_size = max_word_size
    
  def char2id(self,char):
    if char in self.vocab:
      return self.vocab.index(char)
    else:
      return -1
    
  def id2char(self,dictid):
    if dictid != self.eos:
      return self.vocab[dictid]
    else:
      return self.eos_char

  # Word helpers
  def genRandWord(self):
    word_len = np.random.randint(1,self.max_word_size)
    word = [id2char(np.random.randint(1,self.vocab_size)) for _ in range(word_len)]
    return ''.join(word)

  def genRandBatch(self):
      word = self.genRandWord()
      batch = self.word2batch(word)
      rev_batch = self.word2batch(self.reverseWord(word))
      return castInt(batch),castInt(rev_batch)

  # REVERSE HELPERS
  def reverseWord(self,word):
    return word[::-1]

  # BATCH CONVERSIONS
  def batch2word(self,batch):
      return ''.join([self.id2char(i) for i in batch if i != self.eos]) # Skip End of Sequence tag

  def word2batch(self,word):
      batch = [self.char2id(letter) for letter in word] + [self.eos] # Add End of Sequence tag
      return batch

  def id2onehot(self,i):
      oh = np.zeros(self.vocab_size)
      oh[i] = 1
      return oh

  def words2text(self,words):
      text = ''
      for w in words:
          if len(w) > self.max_word_size:
              words.remove(w)
          else:
              text += w + (' ' * (self.max_word_size - len(w)))
      return text        

##  def load(self,hot,words=False):
##      '''Load dat mnist'''
##
##      filename = maybe_download('text8.zip', 31344016)
##
##      text = read_data(filename)
##      print('Data size %d' % len(text))
##
##
##      valid_text = text[:VALID_SIZE]
##      test_text = text[VALID_SIZE:VALID_SIZE+TEST_SIZE]
##      # setting this for testing purposes
##      test_text = 'the quick brown fox ' + test_text[20:]
##      train_text = text[VALID_SIZE+TEST_SIZE:VALID_SIZE+TEST_SIZE+TRAIN_SIZE]
##
##      # Convert each set into batch_size blocks of individual words
##      if words:
##          valid_words = valid_text.split(' ')
##          test_words = test_text.split(' ')
##          train_words = train_text.split(' ')
##
##          print("number of train_words:",len(train_words))
##          print("number of test_words:",len(test_words))
##          print("number of valid_words:",len(valid_words))
##
##          train_text = words2text(train_words)
##          test_text = words2text(test_words)
##          valid_text = words2text(valid_words)
##
##      train_size = len(train_text)
##      test_size = len(test_text)
##      valid_size = len(valid_text)
##
##      print(train_size, train_text[:64])
##      print(test_size, test_text[:64])
##      print(valid_size, valid_text[:64])
##
##      trX = text2onehot(train_text,train_size)
##      teX = text2onehot(test_text,test_size)
##      tvX = text2onehot(valid_text,valid_size)
##      trY = reverseData(trX,words)
##      teY = reverseData(teX,words)
##      tvY = reverseData(tvX,words)
##
##      train_x = castData(trX)
##      train_y = castData(trY)
##      test_x = castData(teX)
##      test_y = castData(teY)
##      valid_x = castData(tvX)
##      valid_y = castData(tvY)
##      if not hot:
##          train_y = T.cast(castData(trY), 'int32')
##          test_y = T.cast(castData(teY), 'int32')
##          valid_y = T.cast(castData(tvY), 'int32')
##      return train_x,train_y,test_x,test_y,valid_x,valid_y
##
