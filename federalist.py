###############################################################
#                        RNN THEANO
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
# LAYERS
from vudu.layer import RecurrentLayer,SoftmaxLayer
# VUDU -- Helpers
from vudu.wordHelpers import WordHelper
from vudu import utils
from vudu.data import loadText

####################################################################################################
# CONSTANTS

# VARIABLES INIT
X_LIST = T.imatrix('x_list')
X = T.ivector('x')
Y_LIST = T.ivector('y_list')
Y = T.iscalar('y')
H = T.dmatrix('hidden_update')

# LOAD DATA
corpus = loadText("federalist.txt")
corpus_len = len(corpus)

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
#   This class handles the set-up for our problem.
#   There are a lot of ways this could be done, but
#   I prefer to have the class defined in the same
#   file as the training loop.
#   It makes it easier to debug and change the
#   structure quickly.
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
        return cost,pred,H

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
            new_m = m + (g * g)
            # Here's where the update list mentioned in
            # init comes into play.
            updates.append((m,new_m))
            updates.append((p, p - ((lr * g) / T.sqrt(new_m + 1e-8))))
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

# This is probably the weirdest part of the whole program.
# Theano has a lot of different optimization tricks and
# formats to accomodate GPU acceleration.
# Unfortunately not all of them are easy to read or understand.
# Instead of for-loops Theano uses the scan function.
#   The scan function takes in the function you wish to loop over - fn,
#   the variables you want to loop over - sequences,
#   and a mysterious list dictating how updates are applied - outputs_info.
#       This outputs_info is straightfoward once you understand what's
#       going on, but at first it looks like witchcraft.
#       The elements of the list corresponds to the outputs of fn.
#       outputs_info dictates whether a value is recurrent or not,
#       ie, whether or not the previous value is accessible
#       or not.  In our case we don't want to compound error or prediction,
#       but we do want to have access to the previous hidden state
#       because that's required for our recurrent architecture.
#       So we pass None for our first two arguments (cost,pred)
#       and a dictionary allowing access to the previous
#       value of H (taps[-1])
outputs_info=[None,None,dict(initial=H, taps=[-1])]

# A final wrinkle in this scan function is that
# it returns 2 outputs.
# 1 is the outputs from the scan loop - results
# and the second is a corresponding list of updates - updates.
#   Note: this is not the same updates list that our class uses.
# We don't need the updates, only the results.
# So we access only the first element of the return - [0].
#
# I've done a lot of complaining about the scan function,
# but it's actually very powerful. Not only can it handle
# the complex task of looping and accessing values recurrently
# flexibly, but it's smart enough to keep track of our inputs.
# Notice we pass in sequences (X_LIST,Y_LIST), but
# our function (calc_cost) only takes individual values.
# scan loops over our lists and takes the corresponding
# value from each list and feeds it into our calc_cost.
# Notably it uses the sequence values first, then looks
# to outputs_info for remaining variables.
# Thus H follows X,Y in our calc_cost definition.
scan_costs,y_preds,hiddens = theano.scan(fn=rnn.calc_cost,
                              outputs_info=outputs_info,
                              sequences=[X_LIST,Y_LIST]
                            )[0] # only need the results, not the updates

# Scan returns a list of our costs (one for every element of X_LIST)
# so we need to sum that list to get our total cost.
scan_cost = T.sum(scan_costs)
# Theano can perform some very clever optimization at compile time
# and one of those is evidenced here.
# By accessing the final element of the list of hidden states from our scan
# loop Theano knows it doesn't need to save the other values during scans.
hidden_state = hiddens[-1]
# We perform the same trick for the prediction list
# since we only need the final prediction.
y_pred = y_preds[-1]


# Here we create the update list mentioned in RNN.__init__
# It's the return from our optimizer.
updates = rnn.Adagrad(scan_cost,params,memory_params)
# This is where the magic happens.
#   We define a theano function to take in a list of X's - one-hot encoded letters,
#   a list of Y's - letters (index of vocab),
#   and our hidden layer - H.
# You may be wondering why H is passed into this function when it really should be
#
back_prop = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=[scan_cost,hidden_state], updates=updates)

#grads = T.grad(cost=scan_cost, wrt=params)
#test_grads  = theano.function(inputs=[X_LIST,Y_LIST,H], outputs=grads, updates=None, allow_input_downcast=True)


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
        predictTest()
        print("Completed iteration:",n,"Cost: ",smooth_loss,"Learning Rate:")

    p += seq_length
    n += 1

print("Training complete")
predictTest()
