import numpy as np
from logger import logger
from abc import ABC, abstractmethod
from utils import *

'''The abstract layer class defines what must be implemented for a 'layer' to be valid within a ML model.
It requires the existece of forward and backwards, which are used in the forward pass and for calculating gradients as well.'''
class Layer(ABC):
    def __init__(self):
        self.name = "Abstract Layer or Name Not Implemented"
    
    @abstractmethod
    def forward(self):
        logger.info(f'Running forward pass on {self.name}')
    
    @abstractmethod
    def backward(self):
        logger.info(f'Running backward pass on {self.name}')


'''The relu layer simply rectifies any nonzero values on its input. This class exists to handle the 
backpropogation of gradients more granularly for just the ReLU layer.'''
class ReLULayer(Layer):
    def __init__(self, name= None):
        super().__init__()
        self.name = name or 'layer_ReLU'
            
    def forward(self, input_data):
        super().forward()
        self.input = np.copy(input_data)
        out = np.copy(self.input)
        out[out < 0] = 0
        return out
    def backward(self, upstream_gradient):
        super().backward()
        return (self.input > 0).astype(np.float64) * upstream_gradient

'''The fully connected layer is a singular layer of the familiar multi-layer-perceptron model.
This layer is the basis for many of the functions of a transformer.
For example, calculating the key, query, and value matrices involves
Running a fully connected network over every input token'''

class FullyConnectedLayer(Layer):
    # Initialize weights matrix of shape (output_nodes, input_nodes)
    # Sets up forward pass to be self.weights @ input_values
    # Where @ is numpy matmul
    def __init__(self, output_nodes: int, input_nodes: int, name= None, starting_weights_in = None):
        super().__init__()
        self.name = name or 'layer_fc'
        self.reLU = ReLULayer(name= (self.name + '_ReLU'))
        weights_shape = (output_nodes, input_nodes)
        self.initialize_weights(weights_shape, starting_weights=starting_weights_in)
        
    def initialize_weights(self, weights_shape, starting_weights = None):
        # Initialize weights
        self.weights = None
        if np.any(starting_weights):
            if(starting_weights.shape == weights_shape):
                self.weights = np.copy(starting_weights)
            else:
                logger.error('statrting_weights does not match given input and output shape. Using defualt')
                self.weights = np.ones(weights_shape)
        else:
            # TODO: use better random initialization
            self.weights = np.ones(weights_shape)
    
    def forward(self, input_data):
        super().forward()
        self.input = np.copy(input_data)
        if input_data.shape[0] != self.weights.shape[-1]:
            logger.error(f"""input to {self.name} forward() call does not match weights shape.\n
                        weights shape: {self.weights.shape} input shape: {input_data.shape}""")
            exit(1)
        forward_result = self.weights @ input_data
        return self.reLU.forward(forward_result)
    def backward(self, upstream_gradient):
        super().backward()
        reLU_gradient = self.reLU.backward(upstream_gradient)
        return reLU_gradient[:,np.newaxis] * self.input[:, np.newaxis].T

