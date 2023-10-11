import numpy as np
import logging
from abc import ABC, abstractmethod

'''The abstract layer class defines what must be implemented for a 'layer' to be valid within a ML model.
It requires the existece of forward and backwards, which are used in the forward pass and for calculating gradients as well.'''
class Layer(ABC):
    def __init__(self):
        self.name = "layer_unspecified"
    
    @abstractmethod
    def forward(self):
        logging.debug(f'Running forward pass on {self.name}')
    
    @abstractmethod
    def backward(self):
        logging.debug(f'Running backward pass on {self.name}')

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
        self.I = data_in.shape[0]
        scaled = np.divide(data_in, np.sqrt(self.scale))
        shifted = np.exp(scaled - np.max(scaled, axis=self.axis, keepdims=True))
        # Save a copy for the gradient
        softmax = np.divide(shifted, np.sum(shifted, axis=self.axis, keepdims=True))
        self.output = np.copy(softmax)
        return softmax


    def backward(self, upstream_gradient):
        super().backward()
        # Find the jacobian 

        # The jacobian is diagonally symmetric
        added_axis = self.output[:, np.newaxis]
        num_cols = self.output.shape[1]
        #TODO: investigate
        #jacobian = np.repeat(added_axis, num_cols, axis=1)
        jacobian_matrix = np.repeat(added_axis, num_cols, axis=1)
        jacobian_matrix *= -1 * self.output.reshape(self.output.shape[0], self.output.shape[1], 1)
        diagonals = (self.output * (1 - self.output))
        jacobian_matrix[:, np.arange(num_cols), np.arange(num_cols)] = diagonals


        jacobian_matrix *= (1.0 / self.scale)
        # Find the gradient
        gradient = np.sum(upstream_gradient[:, np.newaxis, :] * jacobian_matrix, axis=2)
        logging.debug(f'Softmax gradient:\n{gradient}')
        return gradient


'''Calculates softmax loss with one hot encoding.
see https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function for more info'''
#TODO: not sure if necessary right now
class SoftmaxClasses(Layer):
    def __init__(self, name='softmax_classifier'):
        super().__init__()
    
    def get_softmax(self, input_data):
        self.input = input_data
        m = np.max(input_data, axis=1, keepdims=True)
        self.softmax = np.exp(input_data-m) / np.sum(np.exp(input_data-m), axis=1, keepdims=True)
        return self.softmax
    
    def forward(self, input_data, ground_truth):
        super().forward()
        s = self.get_softmax(input_data)
        s_correct = s[np.arange(len(input_data)), ground_truth].reshape((len(input_data),1))
        return (-1.0/len(input_data)) * np.sum(np.log(s_correct))
    
    def backward(self, ground_truth):
        super().backward()
        y_one_hot = np.zeros(self.input.shape) 
        y_one_hot[np.arange(len(self.input)), ground_truth]= 1
        return (1.0/len(self.input))*(self.softmax - y_one_hot)




class FullyConnectedLayer(Layer):
    '''
    The fully connected layer is a singular layer of the familiar multi-layer-perceptron model,
    but modified to handle a batch of input data.
    '''

    def __init__(self, num_input_nodes: int, num_output_nodes: int, name= 'layer_fc', starting_weights_in= None, learning_rate=0.001):
        '''
        Input:
        Takes argument num_input_nodes in the input shape and num_output_nodes is the output shape. Name is a string
        which is the name of the layer. starting_weights_in is a matrix which defines the initial weights.


        Modifies:
        Sets self.name to name
        Sets self.weights_shape to (num_input_nodes, num_output_nodes)
        Sets self.weights to starting_weights if specified, or to np.ones(self.weights_shape) otherwise.

        Output:
        None
        '''
        super().__init__()
        self.name = name
        self.weights = None
        self.set_learning_rate(learning_rate)
        # self.reLU = ReLULayer(name= (self.name + '_ReLU'))
        self.weights_shape = (num_input_nodes, num_output_nodes)
        if not np.any(starting_weights_in):
            self.set_weights(starting_weights=np.random.rand(*(self.weights_shape)))
        else:
            self.set_weights(starting_weights=starting_weights_in)


    def set_learning_rate(self, learning_rate):
        '''
        Input:
        New learning rate to be used

        Modifies:
        Sets self.learning_rate to learning_rate
        '''
        self.learning_rate = learning_rate


    def set_weights(self, starting_weights):
        '''
        Input:
        Starting weight of shape self.weights_shape. If shape does not match, the program exits and logs an error with shape
        mismatch info.

        Modifies:
        Sets self.weights to starting_weights
        '''
        # Initialize weights
        if(starting_weights.shape != self.weights_shape):
            logging.error('starting_weights does not match given input and output shape.')
            exit(1)
        self.weights = np.copy(starting_weights)
    

    def get_weights(self):
        '''
        Output:
        Returns a copy of the weights matrix.
        '''
        return np.copy(self.weights)
    

    def forward(self, tokens_in):
        '''
        Given an matrix W of shape D x N, forward calculates tokens_in @ W.
        Therefore every column i in w represents the weights for output of the ith token of each sequence.

        Input:
        tokens_in is an I x D matrix, where I is the number of sequences present, and D is the number of tokens in a sequence

        Modifies:
        Sets self.input to a copy of tokens_in for use in backwards pass

        Output:
        I x N matrix, where I is the number of sequences present, and D is the numeber of tokens in a sequence.
        Note: In the query, key and value matrices,N == D, so the input and output are of the same shape.
        '''
        super().forward()
        logging.debug(f'input: \n{tokens_in}')
        self.input = np.copy(tokens_in)
        if tokens_in.shape[-1] != self.weights.shape[0]:
            logging.error(f"""input to {self.name} forward() call does not match weights shape.\n
                        weights shape: {self.weights.shape} input shape: {tokens_in.shape}""")
            exit(1)
        forward_result = tokens_in @ self.weights
        logging.debug(f'output: \n{forward_result}')
        return forward_result
        # return self.reLU.forward(forward_result)

    def backward(self, upstream_gradient, update_weights=True):
        '''
        Input:
        Takes in upstream_gradient as a matrix of shape I x N, where I = self.input[0], 
        N = forward_result.shape[1] = num_output_nodes


        Modifies:
        Calculates gradient, then adds -learning_rate*gradient to self.weights matrix:
        Self.weights is of shape D x N. The forward pass was calculated by input @ self.weights.
        The gradient with respect to the weight matrix W is self.input.T @ upstream_gradient.
        The gradient with respect to the inputs is upstream_gradient @ self.weights.T.


        Output:
        Calculates the gradient for the current layer and returns the result.
        Resulting gradient should be the same shape as self.weights_shape
        '''
        super().backward()
        if not np.any(self.input):
            logging.error(f"""backward pass called without running forwards pass first!""")
            exit(1)
        
        logging.debug(f'upstream gradient:\n{upstream_gradient}')
        # reLU_gradient = self.reLU.backward(upstream_gradient)
        weight_gradients = self.input.T @ upstream_gradient
        input_gradients = upstream_gradient @ self.weights.T
        logging.debug(f'local weight_gradients:\n{weight_gradients}')
        
        if update_weights:
            self.weights -= self.learning_rate * weight_gradients
        return input_gradients



class LayerNorm(Layer):
    def __init__(self, learning_rate=0.01, name='layernorm_layer'):
        self.name = name
        self.eps = 0.001
        self.beta = 0
        self.gamma = 1
        self.learning_rate = 0.01
    
    def forward(self, input_data):
        super().forward()
        self.input = input_data
        self.N = np.size(input_data)
        self.mu = 1. / self.N * np.sum(input_data)
        self.xmu = input_data - self.mu
        self.xmu_sq = np.power(self.xmu, 2)
        self.variance = 1./self.N * np.sum(self.xmu_sq)
        self.sqrt_var = np.sqrt(self.variance + self.eps)
        self.inverted_variance = 1. / self.sqrt_var
        self.normalized = self.xmu * self.inverted_variance
        self.gammax = self.gamma * self.normalized
        self.out = self.gammax + self.beta
        return self.out
    
    # Trying to calculate the LayerNorm gradient by hand was a horrible idea. so lets do it using a computational graph approach.
    # Adapted from: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    def backward(self, upstream_grad):
        super().backward()
        d_beta = upstream_grad * np.sum(self.gammax)
        self.beta -= d_beta * self.learning_rate

        d_gammax = upstream_grad
        d_gamma = np.sum(self.normalized * d_gammax)
        self.gamma -= d_gamma * self.learning_rate

        d_normalized = self.gamma * d_gamma

        d_inverted_variance = np.sum(self.xmu * d_normalized)
        d_xmu = self.inverted_variance * d_normalized

        d_sqrt_var = -1./(self.sqrt_var ** 2) * d_inverted_variance

        d_variance = (0.5 * 1./self.sqrt_var) * d_sqrt_var
        
        d_xmu_sq = 1./self.N * np.ones(upstream_grad.shape) * d_variance

        d_xmu = 2.*self.xmu * d_xmu_sq

        d_mu = -1. * d_xmu

        d_input_data = 1. / self.N * d_mu

        return d_input_data
        
