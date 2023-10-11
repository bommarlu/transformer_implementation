import sys
import logging
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from baselayers import *


def print_weights(network):
    # TODO: add get_weights as mandatory in Layer interface
    print(f'weights: \n{network.get_weights()}')

def test_forward_name():
    new_layer = FullyConnectedLayer(4, 3, name='my_fc_layer')
    new_layer.forward(np.array([[2,1,1,1]]))


def test_set_get_weights():
    weights = np.array([[-1, -1, -1, -1],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],]).T
    new_layer = FullyConnectedLayer(4, 3)
    print(f'weights: \n{new_layer.get_weights()}')
    new_layer.forward(np.array([[2,1,1,1]]))
    new_layer.set_weights(weights)
    print(f'weights: \n{new_layer.get_weights()}')
    new_layer.forward(np.array([[2,1,1,1]]))    

def test_initial_weights():
    weights = np.array([[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 2],]).T
    new_layer = FullyConnectedLayer(4, 3, starting_weights_in=weights)
    new_layer.set_weights(weights)

def test_backward_fc():
    weights = np.array([[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],]).T
    new_layer = FullyConnectedLayer(4, 3, starting_weights_in=weights, learning_rate=10)
    new_layer.forward(np.array([[1,1,1,1],
                                [2,2,2,3]]))
    upstream_gradient = np.array([[-1,-2,-3],
                                  [-4,-5,-6]])
    new_layer.backward(upstream_gradient)
    print_weights(new_layer)


def test_softmax():
    softmax = SoftmaxLayer(axis=1)
    soft_in = np.array([[0, 1], [1, 1]])
    soft_out =  softmax.forward(soft_in)
    print(soft_out)
    print(np.sum(np.power(soft_out, 2)))
    while np.sum(np.power(soft_out, 2)) > 1:
        soft_out =  softmax.forward(soft_in)
        soft_out_back = softmax.backward(2*soft_out)
        soft_in = soft_in - soft_out_back * 0.5
        print(soft_out)
        print(np.sum(np.power(soft_out, 2)))

def test_fc_with_softmax_random_data():
    new_layer = FullyConnectedLayer(15, 25, learning_rate=1)

    softmax = SoftmaxLayer(axis=1)

    data_in = np.random.rand(20, 15)
    loss = None
    while not loss or loss > 0.001:
        soft_in = new_layer.forward(data_in)
        soft_out = softmax.forward(soft_in)
        
        soft_out_back = softmax.backward(2*soft_out)
        new_layer.backward(soft_out_back)
        print(soft_out)
        print(np.sum(np.power(soft_out, 2)))
        loss = np.sum(np.power(soft_out, 2))

def test_fc_with_softmax_random_weights():
    weights = np.random.rand(4, 2)
    new_layer = FullyConnectedLayer(4, 2, starting_weights_in=weights)

    softmax = SoftmaxLayer(axis=1)

    data_in = np.array([[1,1,1,1],
                        [2,2,2,3]])
    loss = None
    while not loss or loss > 1.001:
        soft_in = new_layer.forward(data_in)
        soft_out = softmax.forward(soft_in)
        
        soft_out_back = softmax.backward(2*soft_out)
        new_layer.backward(soft_out_back)
        soft_in = soft_in - soft_out_back * 0.5
        print(soft_out)
        print(np.sum(np.power(soft_out, 2)))
        loss = np.sum(np.power(soft_out, 2))

def test_layernorm():
    data_in = np.array([[1,1,1,1],
                        [2,2,2,3]])
    layernorm = LayerNorm()
    print(layernorm.forward(data_in))
    print((data_in - np.mean(data_in)) / (np.std(data_in) + 0.000001))



def test_layernorm_backwards():
    data_in = np.array([[1,1,1,1],
                        [2,2,2,100]]).astype(np.float64)
    loss = None
    lr = 0.01
    layernorm = LayerNorm(learning_rate=lr)
    while not loss or loss > 0.001:
        output = layernorm.forward(data_in)
        loss = np.sum(np.power(output, 2))
        data_in -= layernorm.backward(2*output) * lr
        print(loss)
        print(layernorm.forward(data_in))
        print(f'input_data:\n{data_in}')

def test_layernorm_numerical_backwards():
    data_in = np.array([[1,1,1,1],
                        [2,2,2,3]])
    perturbed = data_in + 0.01

    layernorm = LayerNorm()
    original_forward = layernorm.forward(data_in)
    perturbed_forward = layernorm.forward(perturbed)

    grads = layernorm.backward(original_forward)
    print((1 + grads) * original_forward)
    print(perturbed_forward)
# test_set_get_weights()
# test_backward_fc()
# test_softmax()
test_softmax()

# test_layernorm()
# test_layernorm_numerical_backwards()
