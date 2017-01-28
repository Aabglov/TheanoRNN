###############################################################
#                        RNN THEANO
###############################################################

# THEANO
import numpy as np
import theano
import theano.tensor as T
# I/O
import pickle
import os
import sys
# LAYERS
from vudu.layer import ExplainLayer,SoftmaxLayer
from vudu import utils
####################################################################################################
# CONSTANTS

# VARIABLES INIT
X = T.imatrix('x')
Y = T.imatrix('y')

class ToyData:
    def __init__(self,num,dim=5):
        self.data = []
        self.num = num
        self.dim = dim

    def genRand(self):
        return np.random.rand()

    def genRandBinary(self):
        return float(round(self.genRand()))

    def genData(self):
        self.data = []
        for _ in range(self.num):
            c1 = self.genRand()
            c2 = self.genRand()
            c3 = self.genRandBinary()
            c4 = self.genRandBinary()
            c5 = self.genRandBinary()
            # a bunch of random, ie , bullshit
            # conditions to generate a label set
            if c1 > 0.3 and c3 == 1.0 and c5 == 1.0:
                y = 1.0
            elif c2 > 0.2 and c4 == 0:
                y = 1.0
            else:
                y = 0
            row = [c1,c2,c3,c4,c5,y]
            self.data.append(row)
        self.array_data = np.asarray(self.data)

def testToy():
    for i in range(100):
        toy = ToyData(100,5)
        toy.genData()
        data = toy.array_data
        y = data[:,-1]
        print("% positive: {}".format(sum(y)/100.))

########################################  DATA     #########################################################
num = 1000
bullshit = ToyData(num)
bullshit.genData()
totally_real_data = bullshit.array_data
x = totally_real_data[:,:-1]
y = totally_real_data[:,-1].reshape((num,1))
##############################################################################################################



######################################################################
# MODEL AND OPTIMIZER
######################################################################
# Explainer Network class
#   This class handles the set-up for our problem.
#   There are a lot of ways this could be done, but
#   I prefer to have the class defined in the same
#   file as the training loop.
#   It makes it easier to debug and change the
#   structure quickly.
class ExplainNetwork:
    def __init__(self,layer_sizes):
        self.layer_sizes = layer_sizes
        # Hidden layer
        self.hidden_layer = ExplainLayer(layer_sizes[0],layer_sizes[1],'h')
        # Hidden layer
        self.hidden_layer2 = ExplainLayer(layer_sizes[1],layer_sizes[2],'h2')
        # Output Layer
        #   Just a standard softmax layer.
        self.output_layer = ExplainLayer(layer_sizes[2],layer_sizes[3],'o')
        # Add update parameters
        # and memory parameters (for Adagrad)
        self.update_params = self.hidden_layer.update_params + self.hidden_layer2.update_params + self.output_layer.update_params
        self.memory_params = self.hidden_layer.memory_params + self.hidden_layer2.memory_params + self.output_layer.memory_params


    # Our cost function
    #   This function takes an input X
    #   a label Y.
    #   It performs each layer's forward prop function
    #   Then calculates the error of our prediction and the given Y.
    #   It returns the calculated cost, the prediction
    def calc_cost(self,X,Y):
        H = self.hidden_layer.forward_prop(X)
        H2 = self.hidden_layer2.forward_prop(H)
        pred = self.output_layer.forward_prop(H2)
        cost = T.nnet.binary_crossentropy(pred,Y).mean()
        return cost,pred,H2,H

    def predict(self,X):
        H = self.hidden_layer.forward_prop(X)
        H2 = self.hidden_layer2.forward_prop(H)
        pred = self.output_layer.forward_prop(H2)
        return pred,H2,H

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

try:
    nn = utils.load_net('explain')
except:
    nn = ExplainNetwork([5,10,10,1])
    print("created new network")
############################################# BEGIN THEANO FUNCTION DEFINITIONS ###################################
params = nn.update_params
memory_params = nn.memory_params

cost,pred,h2,h = nn.calc_cost(X,Y)
y_pred,ph2,ph = nn.predict(X)

updates = nn.Adagrad(cost,params,memory_params)
back_prop = theano.function(inputs=[X,Y], outputs=[cost,pred,h2,h], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[y_pred,ph2,ph], updates=None, allow_input_downcast=True)
##############################################################################################################


if hasattr(nn,'current_loss'):
    smooth_loss = nn.current_loss
else:
    smooth_loss = 1.0

if hasattr(nn,'iterations'):
    n = nn.iterations
else:
    n = 0

try:
    while True:
        #print("n: {}".format(n))
        bp_cost,bp_pred,bp_h2,bp_h = back_prop(x,y)
        avg_cost = bp_cost
        smooth_loss = smooth_loss * 0.999 + avg_cost * 0.001

        if not n % 1000:
            #predictTest(graph_id)
            print("Completed iteration:",n,"Cost: ",smooth_loss)
            nn.current_loss = smooth_loss

        n += 1
        nn.iterations = n


except KeyboardInterrupt:
    utils.save_net(nn,'explain',n)

print("Training complete")
