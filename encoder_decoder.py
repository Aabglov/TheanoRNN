###############################################################
#                        ENCODER - DECODER
#                           WORD REVERSER
###############################################################
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

# THEANO
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams as RandomStreams

# I/O
import pickle
import timeit
import sys
import zipfile
import string
from six.moves.urllib.request import urlretrieve
import copy
import re

# MATH
import random
from math import e,log,sqrt,isnan

# LAYERS
from vudu.layer import OneHot,EmbedLayer,LSTMLayer,LinearLayer,SoftmaxLayer

# HELPERS
from vudu.wordHelpers import WordHelper
from vudu import utils

# INIT RANDOM
srng = RandomStreams()
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X_LIST = T.vector('x_list')
X = T.scalar('x')

Y_LIST = T.matrix('y_list')
Y = T.vector('y')

# TODO
# Populate this in creation of the RNN

E_S1 = T.matrix('encoder_hidden_state1')
E_H1 = T.matrix('encoder_hidden_output1')

E_S2 = T.matrix('encoder_hidden_state2')
E_H2 = T.matrix('encoder_hidden_output2')

D_S1 = T.matrix('decoder_hidden_state1')
D_H1 = T.matrix('decoder_hidden_output1')

D_S2 = T.matrix('decoder_hidden_state2')
D_H2 = T.matrix('decoder_hidden_output2')

HIDDEN_STATES = {'encoder_layer_1':{'state':E_S1,'output':E_H1},
                 'encoder_layer_2':{'state':E_S2,'output':E_H2},
                 'decoder_layer_1':{'state':D_S1,'output':D_H2},
                 'decoder_layer_2':{'state':D_S2,'output':D_H2}
                 }

NUM_PRED = T.iscalar('number_of_preds')
INIT_PRED = T.matrix('init_pred')

def unravelHiddens(HIDDEN_STATES):
    encoder_hiddens = []
    decoder_hiddens = []
    for k,v in HIDDEN_STATES.items():
        if 'encoder' in k:
            encoder_hiddens.append(v['state'])
            encoder_hiddens.append(v['output'])
        elif 'decoder' in k:
            decoder_hiddens.append(v['state'])
            decoder_hiddens.append(v['output'])
    return encoder_hiddens,decoder_hiddens



# Intiate Word Helpers
vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
try:
    with open('word_helpers_encoder_decoder.pkl','rb') as f:
        wh = pickle.load(f) # use encoding='latin1' if converting from python2 object to python3 instance
    print('loaded previous wordHelper object')
except:
    wh = WordHelper(vocab,max_word_size=6)
    with open('word_helpers_encoder_decoder.pkl','wb+') as f:
            pickle.dump(wh,f)
    print("created new wordHelper object")
    

# BATCHES
batch_size = 1 # MAX_WORD_SIZE
embed_size = 10
seq_length = 8 # average word length
#num_batches = int(corpus_len/seq_length)

# TRAINING PARAMS
n_epochs = 100000
cur_epoch = 0
cur_grad = 0.
use_saved = False


encoder_nodes = [100,100]
decoder_nodes = [100,100]
    
####################################################################################################
# MODEL AND OPTIMIZATION
####################################################################################################

# RMSprop is for NERDS
def Adagrad(cost, params, mem, lr=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p,g,m in zip(params, grads, mem):
        g = T.clip(g,-5.,5)
        new_m = m + (g * g)
        updates.append((m,new_m))
        updates.append((p, p - ((lr * g) / T.sqrt(new_m + 1e-6))))
    return updates

class RNN:
    def __init__(self,vocab_size,embed_size,encoder_layer_sizes,decoder_layer_sizes,batch_size,dropout=None):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        # Input Layer
        self.input_layer = OneHot(vocab_size,batch_size,wh.eos)#EmbedLayer(vocab_size,embed_size,batch_size)
        # Init update parameters
        self.update_params = self.input_layer.update_params
        # Init memory parameters fo Adagrad
        self.memory_params = self.input_layer.memory_params
        self.current_loss = 0
        self.trained_iterations = 0
        # Encoder
        encoder_layer_sizes = [self.input_layer.y] + encoder_layer_sizes
        self.encoder_layer_names = []
        for i in range(len(encoder_layer_sizes)-1):
            name = 'encoder_layer_{}'.format(i+1) # begin names at 1, not 0
            self.encoder_layer_names.append(name)
            hl = LSTMLayer(encoder_layer_sizes[i],
                               encoder_layer_sizes[i+1],
                               batch_size,
                               name)
            setattr(self,name,hl)                
            # Add the update parameters to the rnn class
            self.update_params += hl.update_params
            self.memory_params += hl.memory_params

        # Decoder
        decoder_layer_sizes = [encoder_layer_sizes[-1]] + decoder_layer_sizes
        self.decoder_layer_names = []
        for i in range(len(decoder_layer_sizes)-1):
            name = 'decoder_layer_{}'.format(i+1) # begin names at 1, not 0
            self.decoder_layer_names.append(name)
            hl = LSTMLayer(decoder_layer_sizes[i],
                               decoder_layer_sizes[i+1],
                               batch_size,
                               name)
            setattr(self,name,hl)                
            # Add the update parameters to the rnn class
            self.update_params += hl.update_params
            self.memory_params += hl.memory_params

        # Output Layer
        self.output_layer = SoftmaxLayer(decoder_layer_sizes[-1],vocab_size)
        # Update Parameters - Backprop
        self.update_params += self.output_layer.update_params
        # Memory Parameters for Adagrad
        self.memory_params += self.output_layer.memory_params

    # pass the word into the network to set all the hidden states.
    def encode(self,X,*hiddens):
        hiddens = list(hiddens)
        o = self.input_layer.forward_prop(X)
        # len(hiddens) will always be an even number
        # because it contains the hidden state and hidden
        # output of each layer
        for i in range(len(self.encoder_layer_names)):
            n = self.encoder_layer_names[i]
            # Get the encoder layer
            encoder_layer = getattr(self,n)
            # Determine the indicies of the corresponding hidden states.
            # They will always be passed in order of layer (encoder1, encoder2, decoder 1, decoder 2, ...)
            # with state, then output.
            state = 2 * i # Because there are 2 elements in the hidden list for every 1 layer we double i
            output = state + 1 # By adding 1 we get the element after k, which is always the hidden output
            # Forward Propagate
            hiddens[state],hiddens[output] = encoder_layer.forward_prop(o,hiddens[state],hiddens[output])
            o = hiddens[output]
        return hiddens

    # make predictions after the word has been sent through the
    # entire network.
    # The pred_unused is the input from the sequence we use to kick
    # off the prediction.  We don't actually need a value, just a
    # sequence of same length as our input word so we know how many
    # letters to predict.
    def decode(self,INIT_PRED,*hiddens):
        hiddens = list(hiddens)
        for i in range(len(self.decoder_layer_names)):
            n = self.decoder_layer_names[i]
            # Get the decoder layer
            decoder_layer = getattr(self,n)
            # Determine the indicies of the corresponding hidden states.
            # They will always be passed in order of layer (encoder1, encoder2, decoder 1, decoder 2, ...)
            # with state, then output.
            state = 2 * i # Because there are 2 elements in the hidden list for every 1 layer we double i
            output = state + 1 # By adding 1 we get the element after k, which is always the hidden output
            # Forward Propagate
            hiddens[state],hiddens[output] = decoder_layer.forward_prop(INIT_PRED,hiddens[state],hiddens[output])
        # Get predicton 
        INIT_PRED= self.output_layer.forward_prop(hiddens[output])
        pred = T.cast(T.argmax(INIT_PRED),theano.config.floatX)
        # Put all returns into a list so the scan function
        # doesn't have to decompile multiple lists
        return_list = [pred,INIT_PRED] + hiddens
        return return_list

    def calc_cost(self,pred,Y):
        return T.mean(T.nnet.categorical_crossentropy(pred,Y))
        

try:
    rnn = utils.load_net('encoder_decoder') 
except:
    rnn = RNN(wh.vocab_size,embed_size,encoder_nodes,decoder_nodes,batch_size)
    print("created new network")

params = rnn.update_params
memory_params = rnn.memory_params

# Generate the outputs_info
#   This instructs Theano how to update variables during a scan.
#   WARNING: Things are about to get SUPER ugly up in here.
outputs_info = []
ENCODER_HIDDENS,DECODER_HIDDENS = unravelHiddens(HIDDEN_STATES)
for e in ENCODER_HIDDENS:
    outputs_info.append(dict(initial=e, taps=[-1]))

############################################# BEGIN THEANO FUNCTION DEFINITIONS ###################################
# Encode
#   Here we envoke the mysterious and treacherous power of the theano.scan utility
encoder_hiddens = theano.scan(fn=rnn.encode,
                              outputs_info=outputs_info,
                              sequences=[X_LIST]
                            )[0] # only need the results, not the updates
encoder_output = encoder_hiddens[-1][-1][-1]


# Prediction outputs info has a few extra values to keep track of
# so we initialize it with those values.
preds_info = [None,dict(initial=encoder_output, taps=[-1])]
for d in DECODER_HIDDENS:
    preds_info.append(dict(initial=d, taps=[-1]))
    
decoder_outputs = theano.scan(fn=rnn.decode,
                              outputs_info=preds_info,
                              sequences=None,
                              n_steps=NUM_PRED
                            )[0] # only need the results, not the updates


y_preds = decoder_outputs[0]
id_preds = decoder_outputs[1] 
decoder_hiddens = decoder_outputs[2:] # states1,outputs1,states2,outputs2,states3,outputs3,...

scan_costs = theano.scan(fn=rnn.calc_cost,
                              outputs_info=None,
                              sequences=[y_preds,Y_LIST]
                            )[0] # only need the results, not the updates

scan_cost = T.sum(scan_costs)

updates = Adagrad(scan_cost,params,memory_params)
#forward_prop = theano.function(inputs=[X_LIST,ENCODER_HIDDENS], outputs=[encoder_hiddens], updates=None, allow_input_downcast=True)
back_prop = theano.function(inputs=[X_LIST,ENCODER_HIDDENS,NUM_PRED,DECODER_HIDDENS,Y_LIST], outputs=[scan_cost,encoder_hiddens,decoder_hiddens], updates=updates, allow_input_downcast=True)

#grads = T.grad(cost=scan_cost, wrt=params)
#test_grads  = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=grads, updates=None, allow_input_downcast=True)

y_pred = y_preds[-1]
predict = theano.function(inputs=[X_LIST,ENCODER_HIDDENS,NUM_PRED,DECODER_HIDDENS], outputs=[y_preds,encoder_hiddens,decoder_hiddens], updates=None, allow_input_downcast=True)

#test_hidden = theano.function(inputs=[X_LIST,Y_LIST,S1,H1,S2,H2,S3,H3], outputs=[states,outputs], updates=None, allow_input_downcast=True)

############################################# END THEANO FUNCTION DEFINITIONS ###################################

print("Model initialized, beginning training")

def genInitHiddens(rnn):
    encoder_hiddens = {}
    decoder_hiddens = {}
    for n in rnn.encoder_layer_names:
        e = getattr(rnn,n)
        encoder_hiddens [n] = {}
        encoder_hiddens [n]['state'] = np.zeros(e.hidden_state_shape)
        encoder_hiddens [n]['output'] = np.zeros(e.hidden_output_shape)
    for n in rnn.decoder_layer_names:
        d = getattr(rnn,n)
        decoder_hiddens[n] = {}
        decoder_hiddens[n]['state'] = np.zeros(d.hidden_state_shape)
        decoder_hiddens[n]['output'] = np.zeros(d.hidden_output_shape)
    return encoder_hiddens,decoder_hiddens

def predictTest():
    test_corpus = ['the','quick','brown','fox','jumped']
    output = []
    init_pred = 0
    for _ in range(5):
        # RESET HIDDENS
        encoder_hiddens,decoder_hiddens = genInitHiddens(rnn)
        # Prepare prediction inputs
        word = test_corpus[_]
        init_pred = wh.char2id(wh.eos)
        pred_input = []
        pred_output_UNUSED = []
        for i in range(len(word)):
            pred_input.append(wh.char2id(word[i]))
        predictions,encoder_hiddens,decoder_hiddens = predict(pred_input,encoder_hiddens,len(word),decoder_hiddens)
        for p in predictions:
            letter = wh.id2char(np.random.choice(wh.vocab_indices, p=p.ravel())) #srng.choice(size=(1,),a=wh.vocab_indices,p=p)
            output.append(letter)
        output.append(' ')
    print("prediction:",''.join(output),'true:',' '.join(test_corpus))

if hasattr(rnn,'current_loss'):
    smooth_loss = rnn.current_loss
else:
    smooth_loss = -np.log(1.0/wh.vocab_size)*seq_length

if hasattr(rnn,'trained_iterations'):
    n = rnn.trained_iterations
else:
    n = 0
    

init_pred = 0
try:
    while True:
        # Reset memory
        encoder_hiddens,decoder_hiddens = genInitHiddens(rnn)
        init_pred = wh.char2id(wh.eos)
        c_input = wh.genRandWord()
        c_output = wh.reverseWord(c_input)

        batch_input = []
        batch_output = []
        for j in range(len(c_input)):
            c = c_input[j]
            c2 = c_output[j]
            batch_input.append(wh.char2id(c))
            
            batch_output.append(wh.id2onehot(wh.char2id(c2)))

        # Forward and Back propagate
        loss,encoder_hiddens,decoder_hiddens = back_prop(batch_input,encoder_hiddens,len(batch_input),decoder_hiddens,batch_output)

        # Set the loss and current iteration
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        rnn.current_loss = smooth_loss
        rnn.trained_iterations = n

        # This is a debug statement to catch NaNs (not-a-number)
        if isnan(smooth_loss):
            print("back prop nan detected")
            break
            
        if not n % 100:
            predictTest()
            print("Completed iteration:",n,"Cost: ",smooth_loss)

        if not n % 5000:
            utils.save_net(rnn,'encoder_decoder',n)
        n += 1

# If I exit the program with the keyboard interrupt
# this will save the current model
except KeyboardInterrupt:
    utils.save_net(rnn,'encoder_decoder',n)
        
print("Training complete")

      

