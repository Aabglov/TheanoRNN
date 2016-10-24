import numpy as np
import theano
import theano.tensor as T


X = T.scalar('x')
X_LIST = T.vector('x_list')
Y = T.scalar('y')
F = T.scalar('f')

def floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def init_weights(x,y,name):
    return theano.shared(floatX(np.random.randn(x,y)*0.01),name=name,borrow=True)

class LinearLayer:
    def __init__(self,input_size,output_size,name):
        self.x = input_size
        self.y = output_size
        self.w = init_weights(input_size,output_size,'{}_w'.format(name))
        self.b = init_weights(1,output_size,'{}_b'.format(name))
        # Variables updated through back-prop
        self.update_params = [self.w,self.b]
        
     # Expects saved output from last layer
    def forward_prop(self,X):
        self.pyx = T.dot(X,self.w) + self.b
        return self.pyx

def GradientDescent(cost, params, lr=0.001):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p,g in zip(params, grads):
        g = T.clip(g,-5.,5)
        updates.append((p, p - (lr * g)))
    return updates


class NeuralNetwork:
    def __init__(self):
        # Init update parameters
        self.update_params = []

        # Layer 1
        self.layer1 = LinearLayer(1,10,'layer1')
        self.update_params += self.layer1.update_params

        # Layer 2
        self.layer2 = LinearLayer(10,1,'layer2')
        self.update_params += self.layer2.update_params

    def forward_prop1(self,X):
        x1 = self.layer1.forward_prop(X)
        x2 = self.layer2.forward_prop(x1)
        return x2

    def forward_prop2(self,X,F):
        return T.dot(X,F)

    def calc_cost(self,pred,Y):
        return T.mean((pred - Y) ** 2)

network = NeuralNetwork()

outputs_info=[None]
scan_result = theano.scan(fn=network.forward_prop1,
                                        outputs_info=outputs_info,
                                        sequences=[X_LIST]
                                        )[0]

final_result = scan_result[-1]
pred = network.forward_prop2(X,final_result)

cost = network.calc_cost(pred,Y)
updates = GradientDescent(cost,network.update_params)

test = theano.function(inputs=[X_LIST,X,Y], outputs=[cost,final_result,pred],updates=updates, allow_input_downcast=True)
pred = theano.function(inputs=[X_LIST,X], outputs=[final_result,pred],updates=None,allow_input_downcast=True)

# test
for i in range(1000):
    c,f,p = test(np.asarray([1,2,3,4,5,6,7,8,9]),10,11)
    if not i % 10:
        print("iteration {}, F: {}, pred: {}".format(i,f,p))
        
print(test(np.asarray([1,2,3,4,5,6,7,8,9]),10,11)[1])

