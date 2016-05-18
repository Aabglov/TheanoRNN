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
import pickle
import os
import timeit
import sys
import zipfile
import string
#from six.moves import range
from six.moves.urllib.request import urlretrieve

# SCIPY
import random
from math import e,log,sqrt
import scipy.optimize

# LAYERS
from layer import EmbedLayer,LSTMLayer,SoftmaxLayer

# HELPERS
import wordHelpers as wh
import utils

# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X = T.iscalar('x')
E = T.vector('embedded_X')
Y = T.iscalar('y')


# BATCHES
TRAIN_BATCHES = 1000
TEST_BATCHES = int(TRAIN_BATCHES)# * 0.2)
VALID_BATCHES = int(TRAIN_BATCHES * 0.2)
batch_size = 1 # MAX_WORD_SIZE
embed_size = 256

n_epochs = 50000
cur_epoch = 0
cur_grad = 0.
use_saved = False

####################################################################################################
# MODEL AND OPTIMIZATION
######################################################################

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        g = T.clip(g,-5.,5)
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

class RNN:
    def __init__(self,vocab_size,embed_size,output_size,hidden_layer_sizes,batch_size):
        self.embed_layer = EmbedLayer(vocab_size,embed_size,batch_size)
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.update_params = self.embed_layer.update_params
        # The first input of the hidden layers will be embed_size (accepting output from embed matrix)
        # and the last will be output_size
        layer_sizes = [embed_size] + hidden_layer_sizes + [output_size]
        self.hidden_layer_names = []
        for i in range(len(layer_sizes)-1):
            name = 'hidden_layer_{}'.format(i)
            self.hidden_layer_names.append(name)
            setattr(self,name,LSTMLayer(layer_sizes[i],
                                       layer_sizes[i+1],
                                       batch_size,
                                       name))
            # Add the update parameters to the rnn class
            hl = getattr(self,name)
            self.update_params += hl.update_params
        self.softmax_layer = SoftmaxLayer(vocab_size)
        self.update_params += self.softmax_layer.update_params
        
    def forward_prop(self,X):
        o = self.embed_layer.forward_prop(X)
        for layer_name in self.hidden_layer_names:
            hidden_layer = getattr(self,layer_name)
            o = hidden_layer.forward_prop(o)
        self.softmax_layer.forward_prop(o)
        return self.softmax_layer.pyx

    def predict(self):
        self.pred = T.argmax(self.softmax_layer.pyx,axis=1)
        return self.pred

    #def loss(self,Y):
        ## May use a different cost function here
        # return T.mean((self.pred - Y) ** 2)

    
fox = wh.initFox()

nodes = [128,256]

rnn = RNN(wh.vocabulary_size,embed_size,wh.vocabulary_size,nodes,batch_size)

pyx = rnn.forward_prop(X)

y_pred = rnn.predict()

cost = T.mean((y_pred - Y) ** 2)

params = rnn.update_params
update = RMSprop(cost,params,lr=0.01)

train = theano.function(inputs=[X], outputs=pyx, allow_input_downcast=True)
predict = theano.function(inputs=[X,Y], outputs=cost, updates=update, allow_input_downcast=True)

for i in range(10):
    c = train(i)
print('c',c.shape)

predict(wh.EOS,5)

##preds,updates = theano.scan(fn=lambda prior_result, X: predict(prior_result),
##                              outputs_info=T.ones_like(X),
##                              non_sequences=X,
##                              n_steps=max_len)
##params = [theta,states]
##update = RMSprop(cost,params,lr=0.01)
##
##
##start_time = timeit.default_timer()
##print('Optimizing using RMSProp...')
##
##for i in range(10000):
##
##    c = 0.
##    for _ in range(TRAIN_BATCHES):
##        train_input,actual = genRandBatch()
##        for j in range(len(new_batch[0])):
##            # Train 1 letter at a time
##            train(train_input[j])
##        pred = predict(EOS)
##        c += learn(EOS,actual)
##    c /= TRAIN_BATCHES
##    
##    fox_pred = ''
##    for _ in range(4):
##        for f in fox[_][0]:
##            fox_pred += id2char(predict(f)[0])
##        fox_pred += ' '
##    print('Completed iteration ',i,', Cost: ',c,'Fox Validation:',fox_pred)#'Input:',str_input,'True:',str_true,
##    
##    if not i % 100:
##        ss,so,hss,hso,h2ss,h2so,h3ss,h3so = get_states(0)
##        print('saved_output',so[:,0])
##        pickle_save(theta.eval(),'theta.pkl')
##        s = packStates(ss,so,hss,hso,h2ss,h2so,h3ss,h3so)
##        pickle_save(s,'states.pkl')
##
##
##end_time = timeit.default_timer()
##print('The code ran for %.1fs' % ((end_time - start_time)))
##
##
##
##
