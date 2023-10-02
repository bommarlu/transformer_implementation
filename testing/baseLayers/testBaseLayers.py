import sys
import logger as logger
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from baseLayers import *

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


def test_forward_specified_weights():
    weights = np.array([[-1, -1, -1, -1],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],])
    new_layer = FullyConnectedLayer(3, 4, starting_weights_in=weights)
    print(new_layer.forward(np.array([2,1,1,1])))

def test_backward():
    weights = np.array([[-1, -1, -1, -1],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],])
    sample_loss = 1.0
    new_layer = FullyConnectedLayer(3, 4, starting_weights_in=weights)
    print('forward pass: \n', new_layer.forward(np.array([2,1,1,1])))
    print('backward pass: \n', new_layer.backward(upstream_gradient=sample_loss))


# test_forward_name()
test_set_get_weights()
