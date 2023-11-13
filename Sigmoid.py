import numpy as np
class Sigmoid:

    def __init__(self):
        pass
        

    def call(self, input):
        #input ndarray of size mini_batch * num_units
        results = np.array([1/(1+np.exp(-x)) for x in input])
        return results    
    
    def backward(self, pre_act, last_gradient):

        der_sigmoid = self.call(pre_act) * (1-self.call(pre_act))
        new_grad = der_sigmoid * last_gradient

        return new_grad

