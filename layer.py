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
F = T.vector('f')

# RANDOM INIT
def init_weights(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)


class EmbedLayer:
    def __init__(self,vocab_size,embed_size,batch_size):
        self.x = vocab_size+1 # Note we use vocab_size + 1 to account for the EOS (End-of-Sequence) Symbol
        self.y = embed_size
        self.batch_size = batch_size
        self.embed_matrix = init_weights(self.x,self.y,'embed')
        # Variables updated through back-prop
        self.update_params = [self.embed_matrix]
        
    def forward_prop(self,X):
        self.embed = self.embed_matrix[X].reshape((self.batch_size,self.y))
        return self.embed

class LSTMLayerOld:
    def __init__(self,input_size,output_size,batch_size,name):
        self.x = input_size
        self.y = output_size
        # LSTM cell weights
        self.x_all = init_weights(input_size,output_size*4,'{}_x_all'.format(name))
        self.m_all = init_weights(output_size,output_size*4,'{}_m_all'.format(name))
        
        self.ib = init_weights(1,output_size,'{}_ib'.format(name))
        self.fb = init_weights(1,output_size,'{}_fb'.format(name))
        self.cb = init_weights(1,output_size,'{}_cb'.format(name))
        self.ob = init_weights(1,output_size,'{}_ob'.format(name))
        # Saved state and output
        self.saved_state = init_weights(batch_size,output_size,'{}_ss'.format(name))
        self.saved_output = init_weights(batch_size,output_size,'{}_so'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.x_all,self.m_all,self.ib,self.fb,self.cb,self.ob,self.saved_state,self.saved_output] # Should state be included?

    # Expects embedded input
    def forward_prop(self,F):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        n = self.y
        i_mul = T.dot(F,self.x_all)

        # Implementing dropout for regularization via:
        # https://arxiv.org/pdf/1409.2329.pdf
        if self.dropout > 0:
            self.saved_output = dropout(self.saved_output)
            
        o_mul = T.dot(self.saved_output,self.m_all)

        ix_mul = i_mul[:,:n]# tf.matmul(i, ix)
        fx_mul = i_mul[:,n:2*n]# tf.matmul(i, fx)
        cx_mul = i_mul[:,2*n:3*n]# tf.matmul(i, cx)
        ox_mul = i_mul[:,3*n:]# tf.matmul(i, ox)

        im_mul = o_mul[:,:n] # tf.matmul(o,im)
        fm_mul = o_mul[:,n:2*n] # tf.matmul(o,fm)
        cm_mul = o_mul[:,2*n:3*n] # tf.matmul(o,cm)
        om_mul = o_mul[:,3*n:] # tf.matmul(o,om)

        input_gate = T.nnet.sigmoid(ix_mul + im_mul + self.ib)
        forget_gate = T.nnet.sigmoid(fx_mul + fm_mul + self.fb)
        update = cx_mul + cm_mul + self.cb
        self.saved_state = (forget_gate * self.saved_state) + (input_gate * T.tanh(update))
        output_gate = T.nnet.sigmoid(ox_mul + om_mul + self.ob)
        self.saved_output = output_gate * T.tanh(self.saved_state)
        return self.saved_output

class LSTMLayer:
    def __init__(self,input_size,output_size,batch_size,name):
        self.x = input_size
        self.y = output_size
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
        # Saved state and output
        self.saved_state = init_weights(batch_size,output_size,'{}_ss'.format(name))
        self.saved_output = init_weights(batch_size,output_size,'{}_so'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.wi,self.wf,self.wc,self.wo,self.bi,self.bf,self.bc,self.bo,self.saved_state,self.saved_output] # Should state be included?

    # Expects embedded input
    def forward_prop(self,F):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""

        # Implementing dropout for regularization via:
        # https://arxiv.org/pdf/1409.2329.pdf
        if self.dropout > 0:
            self.saved_output = dropout(self.saved_output)

        # since we use this everywhere we just make it a variable
        inner_concat = T.concatenate([self.saved_output,F],axis=1)
        
        forget_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wf) + self.bf)
        input_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wi)  + self.bi)
        update_gate = T.tanh(T.dot(inner_concat,self.wc) + self.bc)
        output_gate = T.nnet.sigmoid(T.dot(inner_concat,self.wo)+ self.bo)
        
        self.saved_state = (forget_gate * self.saved_state) + (input_gate * update_gate)
        self.saved_output = output_gate * T.tanh(self.saved_state)
        return self.saved_output
       
class SoftmaxLayer:
    def __init__(self,input_size,vocab_size):
        self.x = input_size
        self.y = vocab_size
        self.w = init_weights(input_size,vocab_size,'w')
        self.b = init_weights(1,vocab_size,'b')
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
    
     # Expects saved output from last LSTM layer
    def forward_prop(self,F):
        self.pyx = T.dot(F,self.w) + self.b
        self.pred = T.nnet.softmax(self.pyx)
        return self.pred

class LinearLayer:
    def __init__(self,input_size,output_size):
        self.x = input_size
        self.y = output_size
        self.w = init_weights(input_size,output_size,'w')
        self.b = init_weights(1,output_size,'b')
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
        
     # Expects saved output from last LSTM layer
    def forward_prop(self,F):
        self.pyx = T.dot(F,self.w) + self.b
        return self.pyx

