###############################################################
#                        LBFGS THEANO
#                        No more fucking around
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# I/O
from numpy import genfromtxt
from matplotlib import pyplot as plot
import pickle
import os
import timeit
import sys
import zipfile
import random
import string
#from six.moves import range
from six.moves.urllib.request import urlretrieve

# SCIPY
import random
from math import e,log,sqrt
import scipy.optimize


# INIT RANDOM
srng = RandomStreams()


# I/O
def pickle_save(o,filename):
    with open(filename, 'wb') as f:
        pickle.dump(o,f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        o = pickle.load(f)
    return o

def castData(data):
    return theano.shared(floatX(data),borrow=True)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

url = 'http://mattmahoney.net/dc/'

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

MAX_WORD_SIZE = 20
TOTAL_SIZE = 100000
OFFSET = 1
TRAIN_SIZE = int(TOTAL_SIZE * .8)
TEST_SIZE = int(TOTAL_SIZE * .1)
VALID_SIZE = int(TOTAL_SIZE * .1)
count = 0


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return str(f.read(name))
    f.close()

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

def text2onehot(text,size):
    data = np.zeros(shape=(size,vocabulary_size), dtype=np.float32)
    for i in range(size):
        v = char2id(text[i])
        data[i,v] = 1.
    return data

# REVERSE HELPERS
def reverseWords(string):
  return ' '.join(w[::-1] for w in string.split())

def reverseData(data,words,flip=True):
  """ Create a batch with all the one-hot letters reversed, but the padding
  position maintained"""
  rev = np.zeros(shape=(data.shape[0],), dtype=np.float32)
  s = ''.join([id2char(np.argmax(d)) for d in data])

  if flip:
      if words:
          r = words2text(reverseWords(s).split(' '))
      else:
          r = reverseWords(s)
  else:
      r = s
      
  for i in range(len(r)):
    rev[i] = char2id(r[i])
  return rev

def prepareData(data):
    new = np.zeros(shape=(data.shape[0],), dtype=np.float32)
    s = ''.join([id2char(np.argmax(d)) for d in data])
    for i in range(len(s)):
        new[i] = char2id(s[i])
    return new

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [[id2char(int(c)) for c in p] for p in probabilities]

def characters_onehot(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def ids_onehot(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character id"""
  return np.array([embed2onehot(np.array([c])).reshape((vocabulary_size)) for c in np.argmax(probabilities, 1)])

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = []
  for batch in batches:
    w = ''.join([id2char(int(np.argmax(b))) for b in batch]).strip()
    s.append(w)
  return s

def ids2batch(ids):
    batch = np.zeros((len(ids),vocabulary_size))
    for i in range(len(ids)):
        batch[i,ids[i]] = 1
    return batch

def batch2string(batch):
  """Convert a batch back into its (most likely) string
  representation."""
  w = ''.join([id2char(np.argmax(b)) for b in batch]).strip()
  return w

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

# RANDOM INIT
def init_weights(shape,name):
    return theano.shared(floatX(np.random.randn(*shape)*0.01),name=name,borrow=True)

# PROCESSING HELPERS
def rectify(X):
    return T.maximum(X,0.)

def dropout(X,p=0.):
    if p > 0:
        retain_prob = 1-p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

# THETA PACKING/UNPACKING FOR LBFGS
def packTheta(weights):
    t = np.ravel(weights[0])
    for i in range(1,len(weights)):
        t = np.concatenate((t,np.ravel(weights[i])),axis=1) 
    return t

def unpackTheta(t,shapes):
    prev_ind = 0
    weights = {}
    for k,v in iter(shapes.items()):
        x = v['x']
        y = v['y']
        ind = x * y
        weights[k] =  t[prev_ind:prev_ind+ind].reshape((x,y))
        prev_ind += ind
    return weights

def thetaShape(shapes):
    total_size = 0
    for s in shapes:
        total_size += shapes[s]['x'] * shapes[s]['y']
    return (total_size,)


# MODEL
class NeuralNetwork(object):
    """Dat Multi-layer network"""
    def __init__(self, input, shapes, n1, n2, n3, vs, bn):
        """ Initialize the parameters"""

        # initialize theta 
        ts = thetaShape(shapes)
        self.theta = init_weights(ts,'theta')
        weights = unpackTheta(self.theta,shapes)

        # 2 Layer Neural Network
##        shapes = {'h':{'x':vocabulary_size,'y':100},
##          'h2':{'x':100,'y':100},
##          'o':{'x':100,'y':vocabulary_size}}
##        self.w1 = weights['h']
##        self.w2 = weights['h2']
##        self.w3 = weights['o']
##        # compute vector of class-membership probabilities in symbolic form
##        self.dh = dropout(rectify(T.dot(dropout(input, d_in),self.w1)),d_h)
##        self.dh2 = dropout(rectify(T.dot(self.dh,self.w2)),d_h)
##        self.py_x = T.nnet.softmax(T.dot(self.dh2,self.w3))
##
##        # compute prediction as class whose probability is maximal in
##        # symbolic form
##        self.ph = rectify(T.dot(input,self.w1))
##        self.ph2 = rectify(T.dot(self.ph,self.w2))
##        self.y_pred = T.argmax(T.nnet.softmax(T.dot(self.ph2,self.w3)),axis=1)

        # The packing/unpacking process doesn't presever order
        # have to rearrange them here
        self.x_all = weights['x_all']
        self.m_all= weights['m_all']
        self.ib = weights['ib']
        self.fb = weights['fb']
        self.cb = weights['cb']
        self.ob = weights['ob']
        self.saved_output = init_weights((bn,n1),'saved_output')#weights['saved_output']
        self.saved_state = init_weights((bn,n1),'saved_state') #weights['saved_state']
        # Hidden Cell
        self.h_x_all = weights['h_x_all']
        self.h_m_all = weights['h_m_all']
        self.h_ib = weights['h_ib']
        self.h_fb = weights['h_fb']
        self.h_cb = weights['h_cb']
        self.h_ob = weights['h_ob']
        self.h_saved_output =  init_weights((bn,n2),'h_saved_output')#weights['h_saved_output']
        self.h_saved_state = init_weights((bn,n2),'h_saved_state')#weights['h_saved_state']
        # Hidden Cell 2
        self.h2_x_all = weights['h2_x_all']
        self.h2_m_all = weights['h2_m_all']
        self.h2_ib = weights['h2_ib']
        self.h2_fb = weights['h2_fb']
        self.h2_cb = weights['h2_cb']
        self.h2_ob = weights['h2_ob']
        self.h2_saved_output =  init_weights((bn,n3),'h2_saved_output')#weights['h_saved_output']
        self.h2_saved_state = init_weights((bn,n3),'h2_saved_state')#weights['h_saved_state']
        # Hidden Cell 3
        self.h3_x_all = weights['h3_x_all']
        self.h3_m_all = weights['h3_m_all']
        self.h3_ib = weights['h3_ib']
        self.h3_fb = weights['h3_fb']
        self.h3_cb = weights['h3_cb']
        self.h3_ob = weights['h3_ob']
        self.h3_saved_output =  init_weights((bn,vs),'h3_saved_output')#weights['h_saved_output']
        self.h3_saved_state = init_weights((bn,vs),'h3_saved_state')#weights['h_saved_state']
        # Final Weights
        self.w = weights['w']
        self.b = weights['b']

        # Definition of the cell computation.
        def lstm_cell(i,o,state,n,x_all,m_all,ib,fb,cb,ob):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous state and the gates."""
            i_mul = T.dot(i,x_all)
            o_mul = T.dot(o,m_all)

            ix_mul = i_mul[:,:n]# tf.matmul(i, ix)
            fx_mul = i_mul[:,n:2*n]# tf.matmul(i, fx)
            cx_mul = i_mul[:,2*n:3*n]# tf.matmul(i, cx)
            ox_mul = i_mul[:,3*n:]# tf.matmul(i, ox)

            im_mul = o_mul[:,:n] # tf.matmul(o,im)
            fm_mul = o_mul[:,n:2*n] # tf.matmul(o,fm)
            cm_mul = o_mul[:,2*n:3*n] # tf.matmul(o,cm)
            om_mul = o_mul[:,3*n:] # tf.matmul(o,om)

            input_gate = T.nnet.sigmoid(ix_mul + im_mul + ib)
            forget_gate = T.nnet.sigmoid(fx_mul + fm_mul + fb)
            update = cx_mul + cm_mul + cb
            state = (forget_gate * state) + (input_gate * T.tanh(update))
            output_gate = T.nnet.sigmoid(ox_mul + om_mul + ob)
            return output_gate * T.tanh(state), state
        
        self.saved_output, self.saved_state      = lstm_cell(input,                self.saved_output,    self.saved_state,    n1, self.x_all,    self.m_all,    self.ib,    self.fb,    self.cb,    self.ob)
        self.h_saved_output,self.h_saved_state   = lstm_cell(self.saved_output,    self.h_saved_output,  self.h_saved_state,  n2, self.h_x_all,  self.h_m_all,  self.h_ib,  self.h_fb,  self.h_cb,  self.h_ob)
        self.h2_saved_output,self.h2_saved_state = lstm_cell(self.h_saved_output,  self.h2_saved_output, self.h2_saved_state, n3, self.h2_x_all, self.h2_m_all, self.h2_ib, self.h2_fb, self.h2_cb, self.h2_ob)
        self.h3_saved_output,self.h3_saved_state = lstm_cell(self.h2_saved_output, self.h3_saved_output, self.h3_saved_state, vs, self.h3_x_all, self.h3_m_all, self.h3_ib, self.h3_fb, self.h3_cb, self.h3_ob)
        self.py_x = T.nnet.softmax(T.dot(self.h3_saved_output,self.w) + self.b)
        self.y_pred = T.argmax(self.py_x,axis=1)
        
        # keep track of model input
        self.input = x

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.py_x)[T.arange(y.shape[0]), y])
        #return T.mean(T.nnet.categorical_crossentropy(self.py_x,y))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                  the correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def pred2string(self,y):
        return self.y_pred,y
        #print("BATCH PRED: ",batch2string(ids2batch(self.y_pred.eval())))

#def optimize():
        
# GET DATA
train_set_x,train_set_y,test_set_x,test_set_y,valid_set_x,valid_set_y = load(hot=False,words=True)

# BATCHES
batch_size = MAX_WORD_SIZE#20
num_nodes = 256
num_nodes2 = 256
num_nodes3 = 256
n_epochs = 50000
cur_epoch = 0

n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)
n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

# VARIABLES INIT
x = T.matrix()
#y = T.matrix()
y = T.ivector()
minibatch_offset = T.lscalar()


# SHAPES OF WEIGHT MATRICES
rec_shapes = {'x_all':{'x':vocabulary_size,'y':4*num_nodes},
              'm_all':{'x':num_nodes,'y':4*num_nodes},
              'ib':{'x':1,'y':num_nodes},
              'fb':{'x':1,'y':num_nodes},
              'cb':{'x':1,'y':num_nodes},
              'ob':{'x':1,'y':num_nodes},
              
              # NOT TRAINED
              #'saved_output':{'x':batch_size,'y':num_nodes},
              #'saved_state':{'x':batch_size,'y':num_nodes},
              
              # Hidden Cell
              'h_x_all':{'x':num_nodes,'y':4*num_nodes2},
              'h_m_all':{'x':num_nodes2,'y':4*num_nodes2},
              'h_ib':{'x':1,'y':num_nodes2},
              'h_fb':{'x':1,'y':num_nodes2},
              'h_cb':{'x':1,'y':num_nodes2},
              'h_ob':{'x':1,'y':num_nodes2},

              # NOT TRAINED
              #'h_saved_output':{'x':batch_size,'y':num_nodes2},
              #'h_saved_state':{'x':batch_size,'y':num_nodes2},

               # Hidden Cell 2
              'h2_x_all':{'x':num_nodes2,'y':4*num_nodes3},
              'h2_m_all':{'x':num_nodes3,'y':4*num_nodes3},
              'h2_ib':{'x':1,'y':num_nodes3},
              'h2_fb':{'x':1,'y':num_nodes3},
              'h2_cb':{'x':1,'y':num_nodes3},
              'h2_ob':{'x':1,'y':num_nodes3},

              # NOT TRAINED
              #'h2_saved_output':{'x':batch_size,'y':num_nodes2},
              #'h2_saved_state':{'x':batch_size,'y':num_nodes2},

               # Hidden Cell 3
              'h3_x_all':{'x':num_nodes3,'y':4*vocabulary_size},
              'h3_m_all':{'x':vocabulary_size,'y':4*vocabulary_size},
              'h3_ib':{'x':1,'y':vocabulary_size},
              'h3_fb':{'x':1,'y':vocabulary_size},
              'h3_cb':{'x':1,'y':vocabulary_size},
              'h3_ob':{'x':1,'y':vocabulary_size},

              # NOT TRAINED
              #'h3_saved_output':{'x':batch_size,'y':vocabulary_size},
              #'h3_saved_state':{'x':batch_size,'y':vocabulary_size},
              
              'w':{'x':vocabulary_size,'y':vocabulary_size},
              'b':{'x':1,'y':vocabulary_size},
              }

# Construct dat class
classifier = NeuralNetwork(input=x, shapes=rec_shapes,n1=num_nodes,n2=num_nodes2,n3=num_nodes3, vs=vocabulary_size, bn=batch_size)

# the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
cost = classifier.negative_log_likelihood(y).mean()

# compile a theano function that computes the mistakes that are made by
# the model on a minibatch

test_model = theano.function(
    [minibatch_offset],
    classifier.errors(y),
    givens={
        x: test_set_x[minibatch_offset:minibatch_offset + batch_size],
        y: test_set_y[minibatch_offset:minibatch_offset + batch_size]
    },
    name="test"
)

validate_model = theano.function(
    [minibatch_offset],
    classifier.errors(y),
    givens={
        x: valid_set_x[minibatch_offset: minibatch_offset + batch_size],
        y: valid_set_y[minibatch_offset: minibatch_offset + batch_size]
    },
    name="validate"
)

#  compile a theano function that returns the cost of a minibatch
batch_cost = theano.function(
    [minibatch_offset],
    cost,
    givens={
        x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
        y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
    },
    name="batch_cost"
)

# compile a theano function that returns the gradient of the minibatch
# with respect to theta
batch_grad = theano.function(
    [minibatch_offset],
    T.grad(cost, classifier.theta),
    givens={
        x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
        y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
    },
    name="batch_grad"
)

batch_print = theano.function(
    [minibatch_offset],
    classifier.pred2string(y),
    givens={
        x: test_set_x[minibatch_offset: minibatch_offset + batch_size],
        y: test_set_y[minibatch_offset: minibatch_offset + batch_size]
    },
    name="batch_print"
)

current_state = theano.function(
    [minibatch_offset],
    classifier.saved_output,
    givens={
        x: test_set_x[minibatch_offset: minibatch_offset + batch_size]
    },
    name="current_state"
)


# creates a function that computes the average cost on the training set
def train_fn(theta_value):
    classifier.theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i * batch_size) for i in range(n_train_batches)]
    return np.mean(train_losses)

# creates a function that computes the average gradient of cost with
# respect to theta
def train_fn_grad(theta_value):
    classifier.theta.set_value(theta_value, borrow=True)
    grad = batch_grad(0)
    for i in range(1,n_train_batches):
        grad += batch_grad(i * batch_size)
    return grad / n_train_batches

validation_scores = [np.inf, 0]

# creates the validation function
def callback(theta_value):
    global cur_epoch
    classifier.theta.set_value(theta_value, borrow=True)
    #compute the validation loss
    validation_losses = [validate_model(i * batch_size) for i in range(n_valid_batches)]
    this_validation_loss = np.mean(validation_losses)
    y_pred = []
    y_true = []
    for i in range(4):
        batch_output = batch_print(i*batch_size)
        y_pred.append(batch_output[0])
        y_true.append(batch_output[1])
    str_pred = ' '.join([batch2string(ids2batch(y)) for y in y_pred])
    str_true = ' '.join([batch2string(ids2batch(y)) for y in y_true])
    print(cur_epoch,'validation error %f %%' % (this_validation_loss * 100),"TRUE:",str_true,"PREDICTION:",str_pred)#,"SO:",current_state(0)[:,0])
    cur_epoch += 1
    # check if it is better then best validation score got until now
    if this_validation_loss < validation_scores[0]:
        # if so, replace the old one, and compute the score on the
        # testing dataset
        validation_scores[0] = this_validation_loss
        test_losses = [test_model(i * batch_size)
                       for i in range(n_test_batches)]
        validation_scores[1] = np.mean(test_losses)
    if not cur_epoch % 100:
        pickle_save(theta_value,'theta.pkl')

ts = thetaShape(rec_shapes)
start_time = timeit.default_timer()

print ("Optimizing using LBFGS ...")
best_w_b = scipy.optimize.fmin_l_bfgs_b(
    func=train_fn,
    x0=np.random.randn(*ts)*0.01,
    #x0=pickle_load('theta.pkl')[0],
    fprime=train_fn_grad,
    callback=callback,
    disp=0,
    maxiter=n_epochs,
    pgtol=1e-7
)

##print ("Optimizing using BFGS ...")
##best_w_b = scipy.optimize.fmin_bfgs(
##    f=train_fn,
##    #x0=np.random.randn(*ts)*0.01,
##    x0=pickle_load('theta.pkl')[0],
##    fprime=train_fn_grad,
##    callback=callback,
##    disp=0,
##    maxiter=n_epochs,
##    gtol=1e-7
##)

##print ("Optimizing using CG ...")
##best_w_b = scipy.optimize.fmin_cg(
##    f=train_fn,
##    #x0=np.random.randn(*ts)*0.01,
##    x0=pickle_load('theta.pkl'),
##    fprime=train_fn_grad,
##    callback=callback,
##    disp=0,
##    maxiter=n_epochs,
##)

end_time = timeit.default_timer()
print(
    ( 'Optimization complete with best validation score of %f %%, with '
        'test performance %f %%') % (validation_scores[0] * 100., validation_scores[1] * 100.) )

print('The code ran for %.1fs' % ((end_time - start_time)))



#if __name__ == '__main__':
#    optimize()

