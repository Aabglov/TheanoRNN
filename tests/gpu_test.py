###############################################################
#                        THE FINAL UDACITY PROJECT
#                           WORD REVERSER
###############################################################
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

# THEANO
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams as RandomStreams

# I/O
#import sys
from trees import load_ids_separated

# INIT RANDOM
srng = RandomStreams()

print("VERSION 2")

####################################################################################################
# CONSTANTS
# Python3 and Python2 behave slightly differently.
# This is here so I can keep track of which version I'm running.
#if (sys.version_info > (3, 0)):
#    python3 = True
#else:
#    python3 = False
python3 = False

# VARIABLES INIT
X = T.matrix('x')
Y = T.matrix('y')
    

trX,trY,unique_ids = load_ids_separated(height_normalized=False)

graph_id = unique_ids[2]

# get input/output size dynamically
input_size = trX[graph_id].shape[1]
output_size = trY[graph_id].shape[1]
nodes = [input_size,output_size]

#################################################################################################
def floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def castData(data):
    return floatX(data)#theano.shared(floatX(data),borrow=True)
#################################################################################################

cast_x = {}
if python3:
    for k,v in trX.items():
        cast_x[k] = castData(v)
    cast_y = {}
    for k,v in trY.items():
        cast_y[k] = castData(v)
else:
    for k,v in trX.iteritems():
        cast_x[k] = castData(v)
    cast_y = {}
    for k,v in trY.iteritems():
        cast_y[k] = castData(v)




    
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

##################################################################################
def init_weights(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)

def init_zeros(x,y,name):
    return theano.shared(floatX(np.zeros((x,y))),name=name,borrow=True)

class LinearLayer:
    def __init__(self,input_size,output_size,name):
        self.x = input_size
        self.y = output_size
        self.current_loss = -np.log(1.0/input_size)*8
        self.iterations = 0
        self.w = init_weights(input_size,output_size,'{}_w'.format(name))
        self.b = init_weights(1,output_size,'{}_b'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
        # Used in Adagrad calculation
        self.mw = init_zeros(input_size,output_size,'m{}_w'.format(name))
        self.mb = init_zeros(1,output_size,'m{}_b'.format(name))
        self.memory_params = [self.mw,self.mb]
        
     # Expects saved output from last layer
    def forward_prop(self,X):
        self.pyx = T.nnet.sigmoid(T.dot(X,self.w) + T.tile(self.b,(X.shape[0],1)))
        return self.pyx
##################################################################################

    
class NeuralNetwork:
    def __init__(self,layer_sizes,dropout=None):
        self.input_size = input_size

        # Init update parameters
        self.update_params = []
        # Init memory parameters fo Adagrad
        self.memory_params = []
        
        # Hidden layers
        self.hidden_layer = LinearLayer(layer_sizes[0],
                           layer_sizes[1],
                           "hidden_layer")
                                              
        # Add the update parameters to the class
        self.update_params += self.hidden_layer.update_params
        self.memory_params += self.hidden_layer.memory_params

    def calc_cost(self,X,Y):
        pred = self.hidden_layer.forward_prop(X)
        cost = T.mean((pred - Y) ** 2)
        return cost

    def predict_y(self,X):
        pred = self.hidden_layer.forward_prop(X)
        return pred
        

nn = NeuralNetwork(nodes)
print("created new network")

params = nn.update_params
memory_params = nn.memory_params

cost = nn.calc_cost(X,Y)
y_pred = nn.predict_y(X)

updates = Adagrad(cost,params,memory_params)
back_prop = theano.function(inputs=[X,Y], outputs=[cost], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[y_pred], updates=None, allow_input_downcast=True)


print("Model initialized, beginning training")


try:
    while True:

        avg_cost = 0
        for i in unique_ids:
            avg_cost += back_prop(cast_x[i],cast_y[i])[0]
        avg_cost /= len(unique_ids)

        smooth_loss = smooth_loss * 0.999 + avg_cost * 0.001
        nn.current_loss = smooth_loss
        
        if not n % 100:
            print("Completed iteration:",n,"Cost: ",smooth_loss)

        n += 1
        nn.iterations = n
        
except KeyboardInterrupt:
    # HELPERS
    print("Training complete")

      

