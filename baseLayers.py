import numpy as np
from logger import logger
from abc import ABC, abstractmethod

'''The abstract layer class defines what must be implemented for a 'layer' to be valid within a ML model.
It requires the existece of forward and backwards, which are used in the forward pass and for calculating gradients as well.'''
class Layer(ABC):
    def __init__(self):
        self.name = "layer_unspecified"
    
    @abstractmethod
    def forward(self):
        logger.info(f'Running forward pass on {self.name}')
    
    @abstractmethod
    def backward(self):
        logger.info(f'Running backward pass on {self.name}')

    #TODO: add shape function


'''The relu layer simply rectifies any nonzero values on its input. This class exists to handle the 
backpropogation of gradients more granularly for just the ReLU layer.'''
class ReLULayer(Layer):
    def __init__(self, name= 'layer_ReLU'):
        super().__init__()
        self.name = name
            
    def forward(self, data_in):
        super().forward()
        self.input = np.copy(data_in)
        out = np.copy(self.input)
        out[out < 0] = 0
        return out
    def backward(self, upstream_gradient):
        super().backward()
        return (self.input > 0).astype(np.float64) * upstream_gradient
    

'''Calculates softmax across the given axis'''
class SoftmaxLayer(Layer):
    def __init__(self, name='layer_softmax', axis=0, scale=1):
        super().__init__()
        name += f'_axis_{axis}'
        name += f'_scaledby_{scale}'
        self.name = name
        self.axis = axis
        self.scale = scale
        
    def forward(self, data_in):
        super().forward()
        scaled = np.divide(data_in, np.sqrt(self.scale))
        shifted = np.exp(scaled - np.max(scaled, axis=self.axis, keepdims=True))
        # Save a copy for the gradient
        softmax = np.divide(shifted, np.sum(shifted, axis=self.axis, keepdims=True))
        self.output = np.copy(softmax)
        return softmax


    def backward(self, upstream_gradient):
        super().backward()
        gradient = ((1.0 / self.scale) * self.output * (1 - self.output)) * upstream_gradient
        return gradient


'''Calculates softmax loss across the given axis, with one hot encoding.
see https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function for more info'''
#TODO: not sure if necessary right now
# class SoftmaxLoss():
#     def __init__(self, name='loss_softmax', axis=0, scale=1):
#         super().__init__()
#         # If default name add conditionals
#         if name == 'loss_softmax':
#             name += f'_axis_{axis}'
#             name += f'_scaledby_{scale}'
#         self.axis = axis
#         self.scale = scale
        
#     def calculate_loss(self, input_data):
#         super().forward()
#         scaled = np.divide(input_data, np.sqrt(self.scale))
#         shifted = np.exp(scaled - np.max(scaled, axis=self.axis, keepdims=True))
#         # Save a copy for the gradient
#         softmax = np.divide(shifted, np.sum(shifted, axis=self.axis, keepdims=True))
#         self.output = np.copy(softmax)
#         return softmax


#     def backward(self, upstream_gradient):
#         super().backward()
#         gradient = ((1.0 / self.scale) * self.output * (1 - self.output)) * upstream_gradient



'''The fully connected layer is a singular layer of the familiar multi-layer-perceptron model,
but modified to handle a batch of input data.'''

class FullyConnectedLayer(Layer):
    # Initialize weights matrix of shape (output_nodes, input_nodes)
    # Sets up forward pass to be self.weights @ input_values
    # Where @ is numpy matmul
    def __init__(self, num_input_nodes: int, num_output_nodes: int, name= 'layer_fc', starting_weights_in= None):
        super().__init__()
        self.name = name
        self.weights = None
        # self.reLU = ReLULayer(name= (self.name + '_ReLU'))
        self.weights_shape = (num_input_nodes, num_output_nodes)
        if np.any(not starting_weights_in):
            self.set_weights(starting_weights=np.ones(self.weights_shape))
        else:
            self.set_weights(starting_weights=np.ones(self.weights_shape))
        
    def set_weights(self, starting_weights):
        # Initialize weights
        if(starting_weights.shape != self.weights_shape):
            logger.error('starting_weights does not match given input and output shape.')
            exit(1)
        self.weights = np.copy(starting_weights)
        
    def get_weights(self):
        return np.copy(self.weights)
            # TODO: use better random initialization
            
    

    '''
    Input:
    tokens_in is an I x D matrix, where I is the number of sequences present, and D is the number of tokens in a sequence

    Given an matrix W of shape D x N, forward calculates tokens_in @ W.
    Therefore every column i in w represents the weights for output of the ith token of each sequence.


    Output:
    I x N matrix, where I is the number of sequences present, and D is the numeber of tokens in a sequence.
    Note: In the query, key and value matrices,N == D, so the input and output are of the same shape.

    '''
    def forward(self, tokens_in):
        super().forward()
        logger.info(f'input: \n{tokens_in}')
        self.input = np.copy(tokens_in)
        if tokens_in.shape[-1] != self.weights.shape[0]:
            logger.error(f"""input to {self.name} forward() call does not match weights shape.\n
                        weights shape: {self.weights.shape} input shape: {tokens_in.shape}""")
            exit(1)
        forward_result = tokens_in @ self.weights
        logger.info(f'output: \n{forward_result}')
        return forward_result
        # return self.reLU.forward(forward_result)
    def backward(self, upstream_gradient):
        super().backward()
        # reLU_gradient = self.reLU.backward(upstream_gradient)
        # return upstream_gradient * self.input[:, np.newaxis].T
        # TODO: fix backward

