###############################################################
#                        RNN THEANO
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# I/O
import io
import pickle
import os
import timeit
import sys
import zipfile
import string
from six.moves.urllib.request import urlretrieve
import copy

# MATH
import random
from math import e,log,sqrt

# LAYERS
from layer import OneHot,EmbedLayer,LSTMLayer,LinearLayer,SoftmaxLayer

# HELPERS
from wordHelpers import WordHelper
import utils

# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS
nodes = [128,128]

# VARIABLES INIT
X_LIST = T.ivector('x_list')
X = T.iscalar('x')

Y_LIST = T.imatrix('y_list')
Y = T.ivector('y')

S1 = T.matrix('hidden_state1')
H1 = T.matrix('hidden_update1')

S2 = T.matrix('hidden_state2')
H2 = T.matrix('hidden_update2')


# LOAD DATA
try:
    # Python 3 load
    with io.open('lor.pkl','rb') as f:
        data = pickle.load(f,encoding='utf-8')
    print("Data found, python3, beginning model init...")
    
except:
    try:
        # Python 2 load
        with io.open('lor.pkl','rb') as f:
            data = pickle.load(f)
        print("Data found, python 2, beginning model init...")
    except:
        print("Failed to find data file, reading lyrics...")
        data = u''
##        for path, subdirs, files in os.walk('/Users/keganrabil/Desktop/lor'):
##            for name in files:
##                file_name = os.path.join(path, name)
##                if file_name[-4:] == '.txt':
##                    with io.open(file_name,'r',encoding='utf-8') as f:
##                        data += f.read()

        with io.open('/Users/keganrabil/Desktop/lor/input.txt','r',encoding='utf-8') as f:
            data += f.read()
        with open('lor.pkl','wb+') as f:
            pickle.dump(data,f)

corpus = data#.lower()
corpus_len = len(corpus)
print("data loaded: {}".format(corpus_len))

# Initialize wordhelper functions
vocab = list(set(corpus))

try:
    with open('word_helpers.pkl','rb') as f:
        wh = pickle.load(f) # use encoding='latin1' if converting from python2 object to python3 instance
    print('loaded previous wordHelper object')
except:
    wh = WordHelper(vocab)
    with open('word_helpers.pkl','wb+') as f:
            pickle.dump(wh,f)
    print("created new wordHelper object")
    

# BATCHES
TRAIN_BATCHES = 1000
TEST_BATCHES = int(TRAIN_BATCHES)# * 0.2)
VALID_BATCHES = int(TRAIN_BATCHES * 0.2)
batch_size = 1 # MAX_WORD_SIZE
embed_size = 50
seq_length = 25
num_batches = int(corpus_len/seq_length)

# TRAINING PARAMS
n_epochs = 100000
cur_epoch = 0
cur_grad = 0.
use_saved = False

# Learning Rate
lr = 0.1
    
####################################################################################################
# MODEL AND OPTIMIZATION
####################################################################################################

# RMSprop is for NERDS
def Adagrad(cost, params, mem):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p,g,m in zip(params, grads, mem):
        g = T.clip(g,-5.,5)
        new_m = m + (g * g)
        updates.append((m,new_m))
        updates.append((p, p - ((lr * g) / T.sqrt(new_m + 1e-8))))
    return updates

class RNN:
    def __init__(self,vocab_size,embed_size,hidden_layer_sizes,batch_size,dropout=None):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        # Input Layer
        self.input_layer = EmbedLayer(vocab_size,embed_size,batch_size)
        # Init update parameters
        self.update_params = self.input_layer.update_params
        # Init memory parameters fo Adagrad
        self.memory_params = self.input_layer.memory_params
        
        # Hidden layer
        layer_sizes = [embed_size] + hidden_layer_sizes
        self.hidden_layer_names = []
        for i in range(len(layer_sizes)-1):
            name = 'hidden_layer_{}'.format(i+1) # begin names at 1, not 0
            self.hidden_layer_names.append(name)
            hl = LSTMLayer(layer_sizes[i],
                               layer_sizes[i+1],
                               batch_size,
                               name
                               ,dropout = 0.5
                           )
            setattr(self,name,hl)                
            # Add the update parameters to the rnn class
            self.update_params += hl.update_params
            self.memory_params += hl.memory_params

        # Output Layer
        self.output_layer = SoftmaxLayer(hidden_layer_sizes[-1],vocab_size)
        # Update Parameters - Backprop
        self.update_params += self.output_layer.update_params
        # Memory Parameters for Adagrad
        self.memory_params += self.output_layer.memory_params

    def calc_cost(self,X,Y,S1,H1,S2,H2):
        e = self.input_layer.forward_prop(X)
        S1,H1 = self.hidden_layer_1.forward_prop(e,S1,H1)
        S2,H2 = self.hidden_layer_2.forward_prop(H1,S2,H2)
        pred = self.output_layer.forward_prop(H2)
        cost = T.nnet.categorical_crossentropy(pred,Y).mean()
        return cost,pred,S1,H1,S2,H2

try:
    with open('lor_rnn.pkl','rb') as f:
        rnn = pickle.load(f) # use encoding='latin1' if converting from python2 object to python3 instance
    print('loaded previous network')
except:
    rnn = RNN(wh.vocab_size,embed_size,nodes,batch_size)
    print("created new network")

params = rnn.update_params
memory_params = rnn.memory_params

outputs_info=[None,None,dict(initial=S1, taps=[-1]),dict(initial=H1, taps=[-1]),
                      dict(initial=S2, taps=[-1]),dict(initial=H2, taps=[-1])
                      ]

scan_costs,y_preds,states1,outputs1,states2,outputs2 = theano.scan(fn=rnn.calc_cost,
                              outputs_info=outputs_info,
                              sequences=[X_LIST,Y_LIST]
                            )[0] # only need the results, not the updates

scan_cost = T.sum(scan_costs)
hidden_state1 = states1[-1]
hidden_output1 = outputs1[-1]
hidden_state2 = states2[-1]
hidden_output2 = outputs2[-1]

updates = Adagrad(scan_cost,params,memory_params)
back_prop = theano.function(inputs=[X_LIST,Y_LIST,S1,H1,S2,H2], outputs=[scan_cost,hidden_state1,hidden_output1,hidden_state2,hidden_output2], updates=updates,  allow_input_downcast=True)

#grads = T.grad(cost=scan_cost, wrt=params)
#test_grads  = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=grads, updates=None, allow_input_downcast=True)

y_pred = y_preds[-1]
predict = theano.function(inputs=[X_LIST,Y_LIST,S1,H1,S2,H2], outputs=[y_pred,hidden_state1,hidden_output1,hidden_state2,hidden_output2], updates=None, allow_input_downcast=True)

#test_hidden = theano.function(inputs=[X_LIST,Y_LIST,S,H], outputs=[states,outputs], updates=None, allow_input_downcast=True)

print("Model initialized, beginning training")

def predictTest(n=100):
    header_in = [wh.char2id(l) for l in "To Whom it may concern"]#corpus[0]
    header_out = [wh.id2onehot(wh.char2id(l)) for l in "o Whom it may concern:"]#corpus[0]
    hidden_state1 = np.zeros(rnn.hidden_layer_1.hidden_state_shape,dtype='float32')
    hidden_output1 = np.zeros(rnn.hidden_layer_1.hidden_output_shape,dtype='float32')
    hidden_state2 = np.zeros(rnn.hidden_layer_2.hidden_state_shape,dtype='float32')
    hidden_output2 = np.zeros(rnn.hidden_layer_2.hidden_output_shape,dtype='float32')
    # Feed header through network
    preds,hidden_state1,hidden_output1,hidden_state2,hidden_output2 = predict(header_in,header_out,hidden_state1,hidden_output1,hidden_state2,hidden_output2)
    # Seed will be colon, the last character in our header
    seed = ":"
    output = [seed]
    for _ in range(n):
        pred_input = [wh.char2id(seed)]
        # This value is only used to trigger the calc_cost.
        # It's incorrect, but it doesn't update the parameters to that's okay.
        # Not great, but okay.
        pred_output_UNUSED = [wh.id2onehot(wh.char2id(corpus[0]))] 
        p,hidden_state1,hidden_output1,hidden_state2,hidden_output2 = predict(pred_input,pred_output_UNUSED,hidden_state1,hidden_output1,hidden_state2,hidden_output2)
        # Changed from argmax to random_choice - good for learnin'
        letter = wh.id2char(np.random.choice(range(wh.vocab_size), p=p.ravel()))
        output.append(letter)
        seed = letter
    print("prediction:",''.join(output))

smooth_loss = -np.log(1.0/wh.vocab_size)*seq_length
n = 0
p = 0
try:
    while True:
        if p+seq_length+1 >= corpus_len or n == 0:
            # Reset memory
            hidden_state1 = np.zeros(rnn.hidden_layer_1.hidden_state_shape,dtype='float32')
            hidden_output1 = np.zeros(rnn.hidden_layer_1.hidden_output_shape,dtype='float32')
            hidden_state2 = np.zeros(rnn.hidden_layer_2.hidden_state_shape,dtype='float32')
            hidden_output2 = np.zeros(rnn.hidden_layer_2.hidden_output_shape,dtype='float32')
            p = 0 # go to beginning
        p2 = p + seq_length
        c_input = corpus[p:p2]
        c_output = corpus[p+1:p2+1]
        
        batch_input = []
        batch_output = []
        for j in range(len(c_input)):
            c = c_input[j]
            c2 = c_output[j]
            batch_input.append(wh.char2id(c))
            batch_output.append(wh.id2onehot(wh.char2id(c2)))
            
        loss,hidden_state1,hidden_output1,hidden_state2,hidden_output2 = back_prop(batch_input,batch_output,hidden_state1,hidden_output1,hidden_state2,hidden_output2)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        if not n % 100:
            predictTest()
            print("Completed iteration:",n,"Cost: ",smooth_loss,"Learning Rate:",lr)

        p += seq_length
        n += 1
except KeyboardInterrupt:
    with open('lor_rnn.pkl','wb+') as f:
        pickle.dump(rnn,f)
        


 
        
print("Training complete")
predictTest()
      

