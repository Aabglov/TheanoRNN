# THEANO
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pickle
import random
import zipfile
import os


# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS


# I/O
def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return str(f.read(name))
    f.close()

def pickle_save(o,filename):
    with open(filename, 'wb') as f:
        pickle.dump(o,f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        o = pickle.load(f,encoding='latin1')
    return o

# Pickle wrappers for network loading/saving
def load_net(model_name):
    model_path = os.path.join(os.path.dirname(__file__),'saved_models')
    files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)) and model_name in f]
    largest_n = 0
    for f in files:
        # Parse the iteration value from the file name to get the latest model
        new_n = int(f.split('_rnn_')[1].split('.')[0])
        if new_n > largest_n:
            largest_n = new_n
    with open(os.path.join(model_path,'{}_rnn_{}.pkl').format(model_name,largest_n),'rb') as f: # use encoding='latin1' if converting from python2 object to python3 instance
        rnn = pickle.load(f,encoding='latin1')
    print("Loaded saved model: {} iterations already trained".format(largest_n))
    return rnn
            
def save_net(rnn,model_name,n):
    model_path = os.path.join(os.path.dirname(__file__),'saved_models')
    with open(os.path.join(model_path,'{}_rnn_{}.pkl').format(model_name,n),'wb+') as f:
        pickle.dump(rnn,f)

def castData(data):
    return theano.shared(floatX(data),borrow=True)

def floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def castInt(data):
    return np.asarray(data, dtype='int32')

# RANDOM INIT
def init_weights(shape,name):
    return theano.shared(floatX(np.random.randn(*shape)*0.01),name=name,borrow=True)

# Ortho init -- Stolen from: https://github.com/mila-udem/blocks/blob/master/blocks/initialization.py
# UNTESTED!
def ortho_init_weights(shape,name):
    if shape[0] == shape[1]:
        # For square weight matrices we can simplify the logic
        # and be more exact:
        M = srng.randn(*shape).astype(theano.config.floatX)
        Q, R = np.linalg.qr(M)
        Q = Q * np.sign(np.diag(R))
        return Q

    M1 = srng.randn(shape[0], shape[0]).astype(theano.config.floatX)
    M2 = srng.randn(shape[1], shape[1]).astype(theano.config.floatX)

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    # Correct that NumPy doesn't force diagonal of R to be non-negative
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))
    n_min = min(shape[0], shape[1])
    weight_mat = np.dot(Q1[:, :n_min], Q2[:n_min, :])
    return theano.shared(floatX(weight_mat),name=name,borrow=True)

# PROCESSING HELPERS
def rectify(X):
    return T.maximum(X,0.)

def dropout(X,p=0.):
    if p > 0:
        retain_prob = 1-p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X
