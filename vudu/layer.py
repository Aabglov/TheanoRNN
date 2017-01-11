# THEANO
import numpy as np
import theano
import theano.tensor as T
import pickle
import random
from vudu.utils import floatX,dropout
from theano.tensor.shared_randomstreams import RandomStreams

# SCIPY
import random

X = T.scalar('x')
F = T.vector('f')
H = T.matrix('h')
S = T.matrix('s')
O = T.matrix('o')
S2 = T.matrix('s2')
O2 = T.matrix('o2')
W = T.matrix('w')

# INIT RANDOM
srng = RandomStreams()

# RANDOM INIT
def init_weights(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)

def init_weights_var(x,y,name,scale):
    return theano.shared(floatX(np.random.randn(x,y)*scale),name=name,borrow=True)

def init_zeros(x,y,name):
    return theano.shared(floatX(np.zeros((x,y))),name=name,borrow=True)

# Not technically a layer, but included here
# for symmetry in the RNN class
class OneHot:
    def __init__(self,vocab_size,batch_size,eos):
        self.x = vocab_size
        self.y = vocab_size
        self.eos = eos
        self.batch_size = batch_size
        self.one_hot_matrix = theano.shared(floatX(np.eye(self.x)),name='one_hot',borrow=True)
        # Variables updated through back-prop
        self.update_params = [] # Placeholders
        # Used in Adagrad calculation
        self.memory_params = [] # Placeholders

    def forward_prop(self,X):
        if X == self.eos:
            # This is a special case to kick off predictions.  If the end-of-sequence tag is passed in
            # then we shouldn't propagate anything through the network so we send in a vector of 0's.
            return theano.shared(floatX(np.zeros((self.batch_size,self.y))),name='one_hot_begin_pred',borrow=True)
        else:
            return self.one_hot_matrix[T.cast(X, 'int32')].reshape((self.batch_size,self.y))

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
        self.embed = self.embed_matrix[T.cast(X, 'int32')].reshape((self.batch_size,self.y))
        return self.embed

class LSTMLayer:
    #def __init__(self,input_size,output_size,batch_size,name,dropout=None):
    def __init__(self,input_size,output_size,name,dropout=None):
        self.x = input_size
        self.y = output_size
        #self.batch_size = batch_size
        #self.hidden_state_shape = (batch_size,output_size)
        #self.hidden_output_shape = (batch_size,output_size)
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
        if self.dropout > 0:
            O = dropout(O) # Dropout function lives in utils.py

        # since we use this everywhere we just make it a variable
        #inner_concat = T.concatenate([O,T.reshape(F,((F.shape[0],self.x)))],axis=1)
        inner_concat = T.concatenate([O,F],axis=1)

        forget_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wf) + T.extra_ops.repeat(self.bf, inner_concat.shape[0], axis=0))#self.bf)
        input_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wi)  + T.extra_ops.repeat(self.bi, inner_concat.shape[0], axis=0))#self.bi)
        update_gate = T.tanh(T.dot(inner_concat,self.wc) + T.extra_ops.repeat(self.bc, inner_concat.shape[0], axis=0))#self.bc)
        output_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wo)+ T.extra_ops.repeat(self.bo, inner_concat.shape[0], axis=0))#self.bo)

        S = T.cast((forget_gate * S) + (input_gate * update_gate),theano.config.floatX)
        O = T.cast(output_gate * T.tanh(S),theano.config.floatX)
        return S,O

# Okay, spitballing here, but I think getting my thoughts down on this will
#   be helpful later on when trying to figure out what I'm doing here.
# I've been doing A LOT of work trying to get existing layers to do what I want.
# Problem is I'm not sure there are any.  Linear and traditional neural network
#   layers (sigmoid feed-forward) are out because they're not recurrent.
# GRU, LSTM and plain Recurrent layers depend too much on the concept of time.
#   Each example comes AFTER the previous.  When your data only exists in 2 states
#   (either preparation examples or examples to be predicted)
#   this paradigm doesn't work.  At least I haven't been able to force it to work.
#
# So what do I need?
#
# I need a layer with an internal state, that's obvious.  It's got to be able to
#   have some kind of memory of values we've given it to train.
#   It needs to be less complicated than an LSTM I think.  Ultimately I want
#   the state of this object to be the same whether I give it 50 or 1000 samples
#   from the same function and to be order agnostic.
#   To me this seems to indicate some kind of batch operation.
#   You pass in all your preparation examples at once and the internal state is set.
#   No loops, no scan, just one update and done.
# I also need the layer to be capable of making a single prediction at a time.
#   This will ultimately be used on data where there is a (essentially) random
#   number of values to predict. Sometimes it'll be 10, sometimes it'll be 1.
#   More importantly, because each value is a prediction it needs to be unaware
#   of those values.  In other words, after the layer makes a prediction it needs
#   to NOT UPDATE.  Otherwise the layer will end up reinforcing any incorrect
#   assumptions it makes.
class MagicChristmasLayer:
    def __init__(self,input_size,output_size,name):
        self.x = input_size
        self.y = output_size
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

    def forward_prop(self,F,S):
        # We assume F is a m x n matrix (m rows, n columns)
        # and S is a 1 x o where o is our output size.
        # Our weight matrix (self.w) will be n x o.

        # Resize our bias to be appropriate size (batch_size x o)
        resized_bias = T.extra_ops.repeat(self.bh, F.shape[0], axis=0)
        # Combine our input data (F) with our weight matrix and bias.
        recurrent_gate = T.dot(F,self.wx) #T.nnet.sigmoid(T.dot(F,self.wx))

        # Resize the state value to have batch_size x output_size shape
        weighted_state = T.dot(S,self.wh)
        hidden_state = T.extra_ops.repeat(weighted_state, F.shape[0], axis=0)

        # Combine the recurrent_gate with our resized hidden state
        # Should I use T.tanh on the hidden_state?
        output = T.nnet.sigmoid(recurrent_gate + hidden_state + resized_bias)

        # This will average the values across the batch_size and
        # return a vector of size 1 x o (output_size)
        new_state = T.mean(hidden_state, axis=0)
        new_state = new_state.reshape((1,self.y))
        # Cast the output
        output_cast = T.cast(output,theano.config.floatX)
        return new_state,output_cast


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
        H = T.tanh(T.dot(F,self.wx) + T.dot(H,self.wh) + T.extra_ops.repeat(self.bh, H.shape[0], axis=0) )#self.bh)
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
        self.pyx = T.nnet.sigmoid(T.dot(F,self.w) + T.tile(self.b,(F.shape[0],1)))
        return self.pyx

# REDUCE LAYER
# Takes in 2 inputs (hidden state and hidden output from an LSTM)
# and produces a single output (not recurrent)
class ReduceLayer:
    def __init__(self,input_size,output_size,name):
        self.x = input_size
        self.y = output_size
        self.w1 = init_weights_var(input_size,output_size,'{}_w1'.format(name),0.01)
        self.b1 = init_weights_var(1,output_size,'{}_b1'.format(name),0.01)
        self.w2 = init_weights_var(input_size,output_size,'{}_w2'.format(name),0.01)
        self.b2 = init_weights_var(1,output_size,'{}_b2'.format(name),0.01)
        self.b3 = init_weights(1,output_size,'{}_b3'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.w1,self.b1,self.w2,self.b2,self.b3]
        # Used in Adagrad calculation
        self.mw1 = init_zeros(input_size,output_size,'m{}_w1'.format(name))
        self.mb1 = init_zeros(1,output_size,'m{}_b1'.format(name))
        self.mw2 = init_zeros(input_size,output_size,'{}_w2'.format(name))
        self.mb2 = init_zeros(1,output_size,'m{}_b2'.format(name))
        self.mb3 = init_zeros(1,output_size,'m{}_b3'.format(name))
        self.memory_params = [self.mw1,self.mb1,self.mw2,self.mb2,self.mb3]

     # Expects saved output from last layer
    def forward_prop(self,F,S,O):
        red_one = T.dot(T.tanh(S),self.w1) + self.b1
        red_two = T.dot(T.tanh(O),self.w2) + self.b2
        self.pyx = T.nnet.sigmoid(T.dot(F,red_one)) + T.nnet.sigmoid(T.dot(F,red_two)) + self.b3
        return self.pyx,S,O,self.b3#red_one,red_two,self.b3


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
        self.pyx = (T.dot(F,self.w) + T.tile(self.b,(F.shape[0],1)))#+ self.b)
        self.pred = T.nnet.softmax(self.pyx).ravel()
        return self.pred
