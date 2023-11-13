from mlp_layer import Mlp_layer 
from Sigmoid import Sigmoid
from softmax import Softmax
from cross_entropy import Cross_entropy
class Multiperceptron:

    def __init__(self, nr_layers, arr_units):
        #arr_units is an array with lenght nr_layers and specifying the nr of units for each layer
        layers = []
        for i in range(nr_layers):
            if i == 0:
                layers.append(Mlp_layer(arr_units[i], arr_units[i], Sigmoid()))
            elif i != nr_layers-1:
                layers.append(Mlp_layer(arr_units[i], arr_units[i-1], Sigmoid()))
            else:
                layers.append(Mlp_layer(arr_units[i], arr_units[i-1], Softmax()))    

        self.layers = layers
        self.loss_func = Cross_entropy()

    def forward(self, input):
        #input batch_size * num_units
        for ind, l in enumerate(self.layers):
            if ind == 0:
                l.forward_step(input)
            else:
                l.forward_step(self.layers[ind-1].layer_act)
        prediction = self.layers[-1].layer_act
        return prediction
    

    def backpropagation(self, gradient_loss):
        
        curr_grad = gradient_loss
        for ind, l in enumerate(reversed(self.layers)):               
                curr_grad = l.backward_step(curr_grad)
        
