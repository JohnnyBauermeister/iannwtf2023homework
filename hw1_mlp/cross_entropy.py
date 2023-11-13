import numpy as np
class Cross_entropy:
    
    def __init__(self):
        pass

    def call(self, prediction, target):
        #categorical distribution of prediction and target minibatch*10

        return np.mean(-1 * np.sum(target * np.log(prediction + 1e-10), axis=1, keepdims=True))
    
    
    def backward(self, prediction, target):

        grad = target - prediction

        return np.mean(grad, axis = 0)