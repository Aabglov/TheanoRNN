###############################################################
#                        LBFGS THEANO
#                        No more fucking around
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
import pickle
import random
from utils import floatX

# SCIPY
import random



# I/O
def pickle_save(o,filename):
    with open(filename, 'wb') as f:
        pickle.dump(o,f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        o = pickle.load(f,encoding='latin1')
    return o



# RANDOM INIT
def init_weights(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)


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

class EmbedLayer:
    def __init__(self,vocab_size,embed_size):
        self.x = vocab_size
        self.y = embed_size
        self.embed_matrix = init_weights(vocab_size,embed_size,'embed')
        
    def forward_prop(self,X):
        self.embed = self.embed_matrix[X].reshape((self.batch_size,self.embed_size))

class LSTMLayer:
    def __init__(self,input_size,output_size,batch_size,name):
        self.x = input_size
        self.y = output_size
        # LSTM cell weights
        self.x_all = init_weights(input_size,output_size*4,'{}_x_all'.format(name))
        self.m_all= init_weights(output_size,output_size*4,'{}_m_all'.format(name))
        self.ib = init_weights(1,output_size,'{}_ib'.format(name))
        self.fb = init_weights(1,output_size,'{}_fb'.format(name))
        self.cb = init_weights(1,output_size,'{}_cb'.format(name))
        self.ob = init_weights(1,output_size,'{}_ob'.format(name))
        # Saved state and output
        self.saved_state = init_weights(batch_size,output_size,'{}_ss'.format(name))
        self.saved_output = init_weights(batch_size,output_size,'{}_so'.format(name))

    # Expects embedded input
    def forward_prop(self,X):
        self.saved_output,self.saved_state = lstm_cell(X,
                                                       self.saved_output,
                                                       self.saved_state,
                                                       self.output_size,
                                                       self.x_all,
                                                       self.m_all,
                                                       self.ib,
                                                       self.fb,
                                                       self.cb,
                                                       self.ob)

class SoftmaxLayer:
    def __init__(self,vocab_size):
        self.x = vocab_size
        self.y = 1
        self.w = init_weights(vocab_size,vocab_size,'w')
        self.b = init_weights(1,vocab_size,'b')

     # Expects saved output from last LSTM layer
    def forward_prop(self,X):
        self.pyx = T.nnet.softmax(T.dot(X,w) + b)
        self.pred = T.argmax(pyx,axis=1)
