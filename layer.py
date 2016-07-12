# THEANO
import numpy as np
import theano
import theano.tensor as T
import pickle
import random
from utils import floatX,dropout

# SCIPY
import random

X = T.iscalar('x')
F = T.dvector('f')
H = T.dmatrix('h')
S = T.dmatrix('s')
O = T.dmatrix('o')

# RANDOM INIT
def init_weights(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)

def init_weights_hidden(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*10.),name=name,borrow=True)

def init_zeros(x,y,name):
    return theano.shared(floatX(np.zeros((x,y))),name=name,borrow=True)

# Not technically a layer, but included here
# for symmetry in the RNN class
class OneHot: 
    def __init__(self,vocab_size,batch_size):
        self.x = vocab_size
        self.y = vocab_size
        self.batch_size = batch_size
        self.one_hot_matrix = theano.shared(floatX(np.eye(self.x)),name='one_hot',borrow=True)
        # Variables updated through back-prop
        self.update_params = [] # Placeholders
        # Used in Adagrad calculation
        self.memory_params = [] # Placeholders
        
    def forward_prop(self,X):
        return self.one_hot_matrix[X].reshape((self.batch_size,self.x))

    
class EmbedLayer:
    def __init__(self,vocab_size,embed_size,batch_size):
        self.x = vocab_size
        self.y = embed_size
        self.batch_size = batch_size
        self.embed_matrix = init_weights(self.x,self.y,'embed')
        # Variables updated through back-prop
        self.update_params = [self.embed_matrix]
        # Used in Adagrad calculation
        self.m_embed_matrix = init_zeros(self.x,self.y,'m_embed')
        self.memory_params = [self.m_embed_matrix]
        
    def forward_prop(self,X):
        self.embed = self.embed_matrix[X].reshape((self.batch_size,self.y))
        return self.embed

class LSTMLayer:
    def __init__(self,input_size,output_size,batch_size,name,dropout=None):
        self.x = input_size
        self.y = output_size
        self.batch_size = batch_size
        self.hidden_state_shape = (batch_size,output_size)
        self.hidden_output_shape = (batch_size,output_size)
        if dropout is None:
            self.dropout = 0
        else:
            self.dropout = dropout
        # LSTM cell weights
        self.wi = init_weights(input_size+output_size,output_size,'{}_wi'.format(name))
        self.wf = init_weights(input_size+output_size,output_size,'{}_wf'.format(name))
        self.wc = init_weights(input_size+output_size,output_size,'{}_wc'.format(name))
        self.wo = init_weights(input_size+output_size,output_size,'{}_wo'.format(name))
        # LSTM cell biases
        self.bi = init_weights(1,output_size,'{}_ib'.format(name))
        self.bf = init_weights(1,output_size,'{}_fb'.format(name))
        self.bc = init_weights(1,output_size,'{}_cb'.format(name))
        self.bo = init_weights(1,output_size,'{}_ob'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.wi,self.wf,self.wc,self.wo,self.bi,self.bf,self.bc,self.bo] 
        # Used in Adagrad calculation
        self.mwi = init_zeros(input_size+output_size,output_size,'m_{}_wi'.format(name))
        self.mwf = init_zeros(input_size+output_size,output_size,'m_{}_wf'.format(name))
        self.mwc = init_zeros(input_size+output_size,output_size,'m_{}_wc'.format(name))
        self.mwo = init_zeros(input_size+output_size,output_size,'m_{}_wo'.format(name))
        self.mbi = init_zeros(1,output_size,'m_{}_ib'.format(name))
        self.mbf = init_zeros(1,output_size,'m_{}_fb'.format(name))
        self.mbc = init_zeros(1,output_size,'m_{}_cb'.format(name))
        self.mbo = init_zeros(1,output_size,'m_{}_ob'.format(name))
        self.memory_params = [self.mwi,self.mwf,self.mwc,self.mwo,self.mbi,self.mbf,self.mbc,self.mbo]


    # Expects embedded input
    def forward_prop(self,F,S,O):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""

        # Implementing dropout for regularization via:
        # https://arxiv.org/pdf/1409.2329.pdf
        #if self.dropout > 0:
        #    O = dropout(O)

        # since we use this everywhere we just make it a variable
        inner_concat = T.concatenate([O,T.reshape(F,((self.batch_size,self.x)))],axis=1)
        
        forget_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wf) + self.bf)
        input_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wi)  + self.bi)
        update_gate = T.tanh(T.dot(inner_concat,self.wc) + self.bc)
        output_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wo)+ self.bo)
        
        S = (forget_gate * S) + (input_gate * update_gate)
        O = output_gate * T.tanh(S)
        return S,O

class RecurrentLayer:
    def __init__(self,input_size,output_size,batch_size,name):
        self.x = input_size
        self.y = output_size
        self.batch_size = batch_size
        self.hidden_state_shape = (batch_size,output_size)
        # Weights
        self.wx = init_weights(input_size,output_size,'{}_wx'.format(name))
        self.wh = init_weights(output_size,output_size,'{}_wh'.format(name))
        # Biases - init with 0
        self.bh = init_zeros(1,output_size,'{}_bh'.format(name))
        # Saved state and output
        #self.hidden_state = init_weights(batch_size,output_size,'{}_hs'.format(name))
        #self.reset = init_zeros(batch_size,output_size,'{}_reset'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.wx,self.wh,self.bh] 
        # Used in Adagrad calculation
        self.mwx = init_zeros(input_size,output_size,'m_{}_wx'.format(name))
        self.mwh = init_zeros(output_size,output_size,'m_{}_wh'.format(name))
        self.mbh = init_zeros(1,output_size,'m_{}_bh'.format(name))
        #self.m_hidden_state = init_zeros(batch_size,output_size,'m_{}_hs'.format(name))
        self.memory_params = [self.mwx,self.mwh,self.mbh]
        
    # Expects embedded input
    def forward_prop(self,F,H):
        H = T.tanh(T.dot(F,self.wx) + T.dot(H,self.wh) + self.bh)
        return H

class LinearLayer:
    def __init__(self,input_size,output_size,name):
        self.x = input_size
        self.y = output_size
        self.w = init_weights(input_size,output_size,'{}_w'.format(name))
        self.b = init_weights(1,output_size,'{}_b'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
        # Used in Adagrad calculation
        self.mw = init_zeros(input_size,output_size,'m{}_w'.format(name))
        self.mb = init_zeros(1,output_size,'m{}_b'.format(name))
        self.memory_params = [self.mw,self.mb]
        
     # Expects saved output from last layer
    def forward_prop(self,F):
        self.pyx = T.dot(F,self.w) + self.b
        return self.pyx
      
class SoftmaxLayer:
    def __init__(self,input_size,vocab_size):
        self.x = input_size
        self.y = vocab_size
        self.w = init_weights(input_size,vocab_size,'w')
        self.b = init_zeros(1,vocab_size,'b')
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
        # Used in Adagrad calculation
        self.mw = init_zeros(input_size,vocab_size,'mw')
        self.mb = init_zeros(1,vocab_size,'mb')
        self.memory_params = [self.mw,self.mb]
    
     # Expects saved output from last LSTM layer
    def forward_prop(self,F):
        self.pyx = (T.dot(F,self.w) + self.b)
        self.pred = T.nnet.softmax(self.pyx).ravel()
        return self.pred



