from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from softmax import Softmax
from cross_entropy import Cross_entropy
from MLP import Multiperceptron
import numpy as np

def test_and_rescale_data(inputs, targets):

    #test plotting the input
    """    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, inputs, targets):
        ax.set_axis_off()
        ax.imshow(image.reshape(8,8), cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Number: %i" % label)
    plt.show()  """
 
    #check if input is between -1 and 1
    inputs_rescaled = (inputs - np.reshape(inputs.min(axis=1), (-1,1)))/ np.reshape(inputs.max(axis=1)-inputs.min(axis=1), (-1,1))
    inputs_rescaled = np.float32(inputs_rescaled)
    cond_float = all(el <= 1 and el >= -1 for el in inputs_rescaled.flatten())
    print('all numbers are between -1 and 1 ', cond_float)

    return inputs_rescaled

def one_hot_encode(targets):
    encoded_targets = np.zeros((targets.shape[0], 10))

    for indx, el in enumerate(targets):
        encoded_targets[indx, el] = 1

    return encoded_targets

def shuffled_pairs(inputs, targets, mini_batch_size):
    rand_indx = np.arange(inputs.shape[0])
    np.random.shuffle(rand_indx)

    inputs = inputs[rand_indx,:]
    targets = targets[rand_indx,:]

    ind_x = 0
    while ind_x < inputs.shape[0]-mini_batch_size:
        mini_input = inputs[ind_x:mini_batch_size+ind_x,:]
        mini_targets = targets[ind_x:mini_batch_size+ind_x,:]
        ind_x += mini_batch_size
        yield (mini_input, mini_targets)

def main():
  
    inputs, targets = load_digits(return_X_y = True)

    inputs = test_and_rescale_data(inputs, targets)
    targets = one_hot_encode(targets)
    input_size = inputs.shape[0]
    mini_batch_size = 2

    #build MP
    nr_layers = 3
    units_per_layer = np.array([64,12,10])
    mlp = Multiperceptron(nr_layers, units_per_layer)
    loss_func = Cross_entropy()

    #init training
    epochs = 10
    loss = np.zeros((epochs, int(input_size/mini_batch_size)))

    for e in range(epochs):
        print('epoch: ', e)
        gen_mini_batches = shuffled_pairs(inputs, targets, mini_batch_size)
        for i in range(int(input_size/mini_batch_size)):
            mini_inputs, mini_targets = next(gen_mini_batches)
            prediction = mlp.forward(mini_inputs)
            gradient_loss = loss_func.backward(prediction, mini_targets)
            mlp.backpropagation(gradient_loss)

            loss[e,i] = loss_func.call(prediction, mini_targets).flatten()
            #print('mean loss ', np.mean(loss[e]))

    mean_loss = np.mean(loss, axis = 1)
    print(mean_loss)

    plt.plot(np.arange(0, epochs), mean_loss, label='mean loss over epochs')
    plt.legend()
    plt.show()



    
if __name__ == '__main__':
    main()