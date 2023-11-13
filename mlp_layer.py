import numpy as np
from Sigmoid import Sigmoid

mu = 0.05
class Mlp_layer():

    def __init__(self, n_units, n_inputs, activation_func):
        self.bias = np.zeros(n_units)
        self.W = np.random.normal(scale = 0.2, size = (n_inputs, n_units))
        self.activation_func = activation_func
        self.layer_act = None
        self.layer_pre_act = None
        self.layer_input = None
        self.num_units = n_units

    def forward_step(self, x):
        #input x of size mini_batch * num_input
        self.layer_input = x
        self.layer_pre_act = np.array([(np.dot(point, self.W) + self.bias) for point in self.layer_input])
        self.layer_act = self.activation_func.call(self.layer_pre_act)
        return self.layer_act
    
    def backward_step(self, last_gradient):
        #input*lastgradient
        if isinstance(self.activation_func, Sigmoid):
            last_gradient = np.mean(self.activation_func.backward(self.layer_pre_act, last_gradient), axis = 0)
        
        grad_W = np.zeros((self.layer_input.shape[0], self.W.shape[0], self.W.shape[1]))
        avg_input = self.layer_input #np.mean(self.layer_input, axis = 0)
        for dp in range(self.layer_input.shape[0]): #AVERAGE OF MINIBATCH NEEDED?
            for unit in range(self.num_units):
                for i in range(self.layer_input.shape[1]):
                    grad_W[dp,i,unit] = avg_input[dp,i] * last_gradient[unit]

        grad_W = np.mean(grad_W, axis = 0)
        grad_inp = np.zeros((self.layer_input.shape[0], self.layer_input.shape[1]))
        
        for dp in range(self.layer_input.shape[0]):
            for unit in range(self.num_units):
                for i in range(self.layer_input.shape[1]):
                    grad_inp[dp,i] += self.W[i, unit] * last_gradient[unit] 
        
        self.W = self.W + mu * grad_W
        return grad_inp
    
