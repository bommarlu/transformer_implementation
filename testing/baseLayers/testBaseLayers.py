import sys
import logger as logger
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from baseLayers import *


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
    new_layer = FullyConnectedLayer(4, 3, starting_weights_in=weights)
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

# test_forward_name()
# test_set_get_weights()
# test_backward_fc()

test_softmax()
