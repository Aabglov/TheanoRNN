import numpy as np
import theano
import theano.tensor as T

H = T.matrix('h')
X = T.vector('x')

k = T.concatenate([H,T.reshape(X,((1,5)))],axis=1)

add_test = theano.function(inputs=[X,H], outputs=k, on_unused_input='warn', allow_input_downcast=True)

x = [1] * 5
h = np.ones((1,10)) * 2.
print(add_test(x,h))
