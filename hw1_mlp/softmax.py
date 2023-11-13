import numpy as np
class Softmax:

    def __init__(self):
        pass

    def call(self, input):
        #input of size mini_batch * 10

        results = np.array([np.exp(x)/(np.sum(np.exp(x))) for x in input])

        return results
    
    def backward(self, activation, grad_before):
        

        grad = np.zeros((activation.shape[0], activation.shape[1]))
        for i in range(activation.shape[0]):
            for j in range(activation.shape[1]):
                grad[i,j] = grad_before[i,j] * activation[i,j]*(1-activation[i,j]) 

        return grad

            
