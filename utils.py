###############################################################
#                        LBFGS THEANO
#                        No more fucking around
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pickle
import random

# SCIPY
import random



# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS


# I/O
def pickle_save(o,filename):
    with open(filename, 'wb') as f:
        pickle.dump(o,f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        o = pickle.load(f,encoding='latin1')
    return o

def thetaLoad(theta_val):
    return theano.shared(floatX(theta_val),name='theta',borrow=True)

def castData(data):
    return theano.shared(floatX(data),borrow=True)

def floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def castInt(data):
    return np.asarray(data, dtype='int32')

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
def pack(weights):
    t = weights[0].ravel()
    for i in range(1,len(weights)):
        t = T.concatenate((t,weights[i].ravel())) 
    return t

def unpack(t,shapes):
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
