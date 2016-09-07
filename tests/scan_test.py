import numpy as np
import theano
import theano.tensor as T

H = T.scalar('h')
X = T.scalar('x')
X_LIST = T.vector('x_list')

# define a named function, rather than using lambda
def add(X,H):
    H = H + 1
    x = X + 2
    y = 3 * X
    return x,y,H

outputs_info=[None,None,dict(initial=H, taps=[-1])]
scan_result, scan_updates = theano.scan(fn=add,
                                        outputs_info=outputs_info,
                                        sequences=[X_LIST]
                                        )
#Xs = scan_result[0]
xs = scan_result[0]
ys = scan_result[1]
Hs = scan_result[2]
add_test = theano.function(inputs=[X_LIST,H], outputs=[xs,ys,Hs], on_unused_input='warn')

# test
print(add_test(np.asarray([1,2,3]),1))

##up_to = T.iscalar("up_to")
##
### define a named function, rather than using lambda
##def accumulate_by_adding(sum_to_date,arange_val):
##    return sum_to_date + arange_val
##seq = T.arange(up_to)
##
### An unauthorized implicit downcast from the dtype of 'seq', to that of
### 'T.as_tensor_variable(0)' which is of dtype 'int8' by default would occur
### if this instruction were to be used instead of the next one:
### outputs_info = T.as_tensor_variable(0)
##
##outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
##scan_result, scan_updates = theano.scan(fn=accumulate_by_adding,
##                                        outputs_info=outputs_info,
##                                        sequences=seq)
##triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)
##
### test
##some_num = 6
##print(triangular_sequence(some_num))
#print([n * (n + 1) // 2 for n in range(some_num)])

