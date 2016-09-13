import numpy as np
import theano
import theano.tensor as T

X = T.scalar('x')
X_LIST = T.vector('x_list')
K = T.scalar('k')

class TestObject:
    def __init__(self):
        self.H = T.scalar('h')
    # define a named function, rather than using lambda
    def add(self,X,K):
        K = K + 1
        x = X + K
        y = 3 * X
        return x,y,K

test = TestObject()
outputs_info=[None,None,dict(initial=K, taps=[-1])]
scan_result, scan_updates = theano.scan(fn=test.add,
                                        outputs_info=outputs_info,
                                        sequences=[X_LIST]
                                        )
#Xs = scan_result[0]
xs = scan_result[0]
ys = scan_result[1]
Hs = scan_result[2]
add_test = theano.function(inputs=[X_LIST,K], outputs=[xs,ys,Hs], on_unused_input='warn')

# test
print(add_test(np.asarray([2]),1))

