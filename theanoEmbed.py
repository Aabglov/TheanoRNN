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
from matplotlib import pyplot as plot
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
import wordHelpers
import utils

# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS

# VARIABLES INIT
#X = T.matrix()
#y = T.matrix()
#X = T.ivector()
#Y = T.ivector()
max_len = T.iscalar('max_len')
X = T.iscalar('x')
Y = T.iscalar('y')
ACTUAL = T.ivector('actual')

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

EOS = -1

MAX_WORD_SIZE = 10
# BATCHES
TRAIN_BATCHES = 1000
TEST_BATCHES = int(TRAIN_BATCHES)# * 0.2)
VALID_BATCHES = int(TRAIN_BATCHES * 0.2)
batch_size = 1#MAX_WORD_SIZE#20
embed_size = 256
num_nodes = 256
num_nodes2 = 256
num_nodes3 = 256
vocabulary_size = len(vocab)

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
        self.embed_layer = EmbedLayer(vocab_size,embed_size)
        # The first input of the hidden layers will be embed_size (accepting output from embed matrix)
        # and the last will be output_size
        layer_sizes = [embed_size] + hidden_layer_sizes
        self.hidden_layer_names = []
        for i in range(len(layer_sizes)-1):
            name = 'hidden_layer_{}'.format(i)
            self.hidden_layer_names.append(name)
            setattr(self,name,LSTMLayer(layer_sizes[i],
                                       layer_sizes[i+1],
                                       batch_size,
                                       name))
        self.softmax_layer = SoftmaxLayer(vocab_size)

    def forward_prop(self,X):
        e = self.embed[X].reshape((self.batch_size,self.embed_size))
        # TODO        
        return pred
        
fox = initFox()

# initialize theta 
ts = thetaShape(shapes)
ss = thetaShape(state_shapes)

if use_saved:
    theta = castData(pickle_load('theta_orig.pkl'))
    states = castData(pickle_load('states_orig.pkl'))
else:
    theta = init_weights(ts,'theta')#thetaLoad(pickle_load('theta_orig.pkl'))
    states = init_weights(ss,'states')

y_pred,py,states = model(X,theta,shapes,states,state_shapes)

train = theano.function(inputs=[X], outputs=states, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

preds,updates = theano.scan(fn=lambda prior_result, X: predict(prior_result),
                              outputs_info=T.ones_like(X),
                              non_sequences=X,
                              n_steps=max_len)

cost = T.mean((preds - ACTUAL) ** 2)
params = [theta,states]
update = RMSprop(cost,params,lr=0.01)

learn = theano.function(inputs=[X,ACTUAL], outputs=cost, updates=update, allow_input_downcast=True)

start_time = timeit.default_timer()
print('Optimizing using RMSProp...')

for i in range(10000):

    c = 0.
    for _ in range(TRAIN_BATCHES):
        train_input,actual = genRandBatch()
        for j in range(len(new_batch[0])):
            # Train 1 letter at a time
            train(train_input[j])
        pred = predict(EOS)
        c += learn(EOS,actual)
    c /= TRAIN_BATCHES
    
    fox_pred = ''
    for _ in range(4):
        for f in fox[_][0]:
            fox_pred += id2char(predict(f)[0])
        fox_pred += ' '
    print('Completed iteration ',i,', Cost: ',c,'Fox Validation:',fox_pred)#'Input:',str_input,'True:',str_true,
    
    if not i % 100:
        ss,so,hss,hso,h2ss,h2so,h3ss,h3so = get_states(0)
        print('saved_output',so[:,0])
        pickle_save(theta.eval(),'theta.pkl')
        s = packStates(ss,so,hss,hso,h2ss,h2so,h3ss,h3so)
        pickle_save(s,'states.pkl')


end_time = timeit.default_timer()
print('The code ran for %.1fs' % ((end_time - start_time)))



