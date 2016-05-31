import theano
from theano import tensor as T
import numpy as np
import layer

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

X = T.iscalar()
Y = T.ivector()

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

class model:
    def __init__(self,vocab_size,embed_size,batch_size,hidden_size):
        self.embed_layer = layer.EmbedLayer(vocab_size,embed_size,batch_size)
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_layer = layer.LSTMLayer(hidden_size,vocab_size,batch_size,'hidden_layer_0')
        self.softmax_layer = layer.LinearLayer(vocab_size,1)
        #self.w = layer.init_weights(hidden_size,1,'w')
        self.params = self.embed_layer.update_params + self.hidden_layer.update_params + self.softmax_layer.update_params #+  [self.w]
    def forward_prop(self,X):
        o = self.embed_layer.forward_prop(X)
        o = self.hidden_layer.forward_prop(o)
        o = self.softmax_layer.forward_prop(o)
        return o


net = model(10,25,1,25)
y = net.forward_prop(X)
cost = T.mean(T.sqr(y - Y))
params = net.params#[net.w,net.w2]
updates = RMSprop(cost=cost,params=params,lr=1.0)
test_back_prop = updates[0]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
test_updates = theano.function(inputs=[X,Y], outputs=test_back_prop, allow_input_downcast=True,on_unused_input='warn')

for i in range(10):
    c = 0.
    #for x, y in zip(trX, trY):
    for j in range(25):
        c += train(j % 10,[(j+1)%10])
    print("cost:",c/len(trX))
        
print(net.w.get_value()) #something around 2

