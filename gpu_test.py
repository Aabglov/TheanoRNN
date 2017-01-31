###############################################################
#                        RNN THEANO
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
import os
from vudu.wordHelpers import WordHelper
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X_LIST = T.imatrix('x_list')
X = T.ivector('x')
Y_LIST = T.ivector('y_list')
Y = T.iscalar('y')
H = T.matrix('hidden_update',dtype=theano.config.floatX)

# LOAD DATA
def loadText(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path,"data",filename),'r') as f:
        data = f.read()
    f.close()
    corpus = data#.lower()
    corpus_len = len(corpus)
    print("data loaded: {}".format(corpus_len))
    return corpus
corpus = loadText("federalist.txt")
corpus_len = len(corpus)

def castData(data):
    return T.cast(data,dtype=theano.config.floatX)#theano.shared(floatX(data),borrow=True)

# RANDOM INIT
def init_weights(x,y,name):
    return floatX(np.random.randn(x,y)*0.01) #theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)

def init_zeros(x,y,name):
    return floatX(np.zeros(x,y)) #theano.shared(floatX(np.zeros((x,y))),name=name,borrow=True)

# Initialize wordhelper functions
vocab = list(set(corpus))
wh = WordHelper(vocab)

# BATCHES
batch_size = 1 # Number of letters we pass in at a time -- just 1
nodes = [100] # Number of hidden units in our Recurrent Layer
seq_length = 25 # Arbitrary constant used for printing


######################################################################
# MODEL AND OPTIMIZER
######################################################################
# Recurrent Neural Network class

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
        pyx = (T.dot(F,self.w) + T.tile(self.b,(F.shape[0],1)))#+ self.b)
        pred = T.nnet.softmax(pyx).ravel()
        return castData(pred)

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
        return castData(H)

class RNN:
    def __init__(self,vocab_size,hidden_layer_size,batch_size):
        self.batch_size = batch_size # Number of letters we pass in at a time -- just 1
        self.hidden_size = hidden_layer_size # Number of hidden units in our Recurrent Layer
        self.vocab_size = vocab_size # Number of letters in our vocabulary - 26
        # Hidden layer
        #   This layer is what makes it a recurrent neural network.
        #   The 'h' at the end is the name given to the variable by Theano.
        self.hidden_layer = RecurrentLayer(vocab_size,hidden_layer_size,batch_size,'h')
        # Output Layer
        #   Just a standard softmax layer.
        self.output_layer = SoftmaxLayer(hidden_layer_size,vocab_size)

        # Update Parameters - Backprop
        #   This part is a little weird.
        #   Theano has this weird way of keeping track of which
        #   variables can get updated by functions.
        #   It uses a list of tuples of the variable and the new value.
        #   So it's important to keep track of which variables we
        #   eventually want to be updated in this handy list.
        #   The dictionary mentioned earlier is constructed in
        #   our backprop algorithm - Adagrad.
        self.update_params = self.hidden_layer.update_params + \
                             self.output_layer.update_params
        # Memory Parameters for Adagrad
        #   These variables are essentially copies of the
        #   previous variables used in the Adagrad algorithm.
        #   They also get updated during back prop so they
        #   must be added separately.
        self.memory_params = self.hidden_layer.memory_params + \
                             self.output_layer.memory_params

    # Our cost function
    #   This function takes an input X - letter
    #   a label Y - also a letter
    #   and a hidden state matrix H.
    #   It performs each layer's forward prop function
    #   Then calculates the error of our prediction and the given Y.
    #   It returns the calculated cost, the prediction and the updated hidden state, H.
    def calc_cost(self,X,Y,H):
        H = self.hidden_layer.forward_prop(X,H)
        pred = self.output_layer.forward_prop(H)
        cost = -T.log(pred[Y])
        return  castData(cost),pred,H

    # RMSprop is for NERDS
    #   The Adagrad function is like
    #   gradient descent on steroids.
    #   It converges must faster and
    #   is more stable.
    def Adagrad(self,cost, params, mem, lr=0.1):
        # This line here is nearly the entire
        # reason to use Theano at all.
        # Being able to calculate the gradient
        # of our model automatically saves
        # huge amounts of time when testing models.
        # With this call I can add 3 more layers
        # to the current model and all I need
        # to update are my variables, class init
        # and some of the function calls.
        # I don't have to throw out my previous
        # calculation for gradients of a 1-layer
        # model and start all over.
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p,g,m in zip(params, grads, mem):
            g = T.clip(g,-5.,5)
            new_m = T.cast(m + (g * g),theano.config.floatX)
            # Here's where the update list mentioned in
            # init comes into play.
            updates.append((m,new_m))
            new_p = T.cast( p - ((lr * g) / T.sqrt(new_m + 1e-8)) ,theano.config.floatX)
            updates.append((p, new_p))
        return updates


######################################################################
# FUNCTIONS AND VARIABLES
######################################################################
# Create our class - Recurrent Neural Network
#   Note: this is not an LSTM, just a recurrent net.
rnn = RNN(wh.vocab_size,nodes[0],batch_size)
# Now we can easily grab the variables
# we eventually want to be able to update
# by referencing our newly created class.
params = rnn.update_params
memory_params = rnn.memory_params


outputs_info=[None,None,dict(initial=H, taps=[-1])]
scan_costs,y_preds,hiddens = theano.scan(fn=rnn.calc_cost,
                              outputs_info=outputs_info,
                              sequences=[X_LIST,Y_LIST]
                            )[0] # only need the results, not the updates


scan_cost = T.sum(scan_costs)
hidden_state = hiddens[-1]
y_pred = y_preds[-1]

updates = rnn.Adagrad(scan_cost,params,memory_params)
back_prop = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=[scan_cost,hidden_state], updates=updates, allow_input_downcast=True)

predict = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=[y_pred,hidden_state], updates=None, allow_input_downcast=True)

test_hidden = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=hiddens, updates=None, allow_input_downcast=True)
print("Model initialized, beginning training")

def predictTest():
    seed = corpus[0]
    output = [seed]
    hidden_state = np.zeros(rnn.hidden_layer.hidden_state_shape)
    for _ in range(seq_length*4):
        pred_input = [wh.id2onehot(wh.char2id(seed))]
        pred_output_UNUSED = [wh.char2id(corpus[0])]
        p,hidden_state = predict(pred_input,pred_output_UNUSED,hidden_state)
        letter = wh.id2char(np.random.choice(range(wh.vocab_size), p=p.ravel()))
        output.append(letter)
        seed = letter
    print("prediction:",''.join(output))

smooth_loss = -np.log(1.0/wh.vocab_size)*seq_length
n = 0
p = 0
while True:
    if p+seq_length+1 >= corpus_len or n == 0:
        # Reset memory
        hidden_state = np.zeros(rnn.hidden_layer.hidden_state_shape)
        p = 0 # go to beginning of corpus
    p2 = p + seq_length
    c_input = corpus[p:p2]
    c_output = corpus[p+1:p2+1]

    batch_input = []
    batch_output = []
    for j in range(len(c_input)):
        c = c_input[j]
        c2 = c_output[j]
        batch_input.append(wh.id2onehot(wh.char2id(c)))
        #batch_output.append(wh.id2onehot(wh.char2id(c2)))
        batch_output.append(wh.char2id(c2))

    loss,hidden_state = back_prop(batch_input,batch_output,hidden_state)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if not n % 100:
        #predictTest()
        print("Completed iteration:",n)#,"Cost: ",smooth_loss,"Learning Rate:")

    p += seq_length
    n += 1

print("Training complete")
predictTest()
