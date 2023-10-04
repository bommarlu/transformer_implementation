import numpy as np
import logging
import math
from baseLayers import SoftmaxLayer, FullyConnectedLayer
from baseLayers import *

# def softmax(x: np.array):
#     raised = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return np.divide(raised, np.sum(raised, axis=1, keepdims=True))

# def scaled_softmax(x: np.array):
#     scaled = np.divide(x, np.sqrt(len(x[0])))
#     raised = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
#     return np.divide(raised, np.sum(raised, axis=1, keepdims=True))

'''Calculates attention matrix of size sequence_length_in * sequence_length_in'''
class AttentionLayer(Layer):
    def __init__(self, sequence_length_in: int, token_length_in: int, token_length_out: int, learning_rate=0.001, name='attention_layer'):
        super().__init__()
        self.name=name
        self.token_length_in = token_length_in
        self.token_length_out = token_length_out # For multiple heads, this is a divisor of token_length in, otherwise 1
        self.sequence_length = sequence_length_in 
        self.attention_shape = (self.sequence_length, self.sequence_length)

        self.query_layer = FullyConnectedLayer(num_output_nodes=self.token_length_out, num_input_nodes=self.token_length_in, name='query_layer', learning_rate=learning_rate)
        self.key_layer = FullyConnectedLayer(num_output_nodes=self.token_length_out, num_input_nodes=self.token_length_in, name='key_layer', learning_rate=learning_rate)
        self.attention = None
        self.softmax_layer = SoftmaxLayer(name='attention_softmax_layer', axis=1, scale=np.sqrt(token_length_out))
    
    def set_key_weights(self, key_weights):
        self.key_layer.set_weights(key_weights)
    def get_key_weights(self):
        return self.key_layer.get_weights()
    def set_query_weights(self, query_weights):
        self.query_layer.set_weights(query_weights)
    def get_query_weights(self):
        return self.query_layer.get_weights()
    
    
    def forward(self, tokens_in: np.array):
        super().forward()
        logging.debug(f'Query Weights:\n{self.query_layer.get_weights()}')
        logging.debug(f'Key Weights:\n{self.key_layer.get_weights()}')

        self.queryMatrix = self.query_layer.forward(tokens_in=tokens_in)
        self.keyMatrix = self.key_layer.forward(tokens_in=tokens_in)

        pre_softmax = self.queryMatrix @ self.keyMatrix.T
        logging.debug(f'Attention matrix, pre softmax:\n{pre_softmax}')

        self.attention = self.softmax_layer.forward(data_in=pre_softmax)
        logging.debug(f'Attention matrix:\n{self.attention}')

        return np.copy(self.attention)
    
    def backward(self, upstream_grad: float):
        super().backward()
        softmax_grad = self.softmax_layer.backward(upstream_grad)
        query_grad = softmax_grad @ self.keyMatrix
        key_grad = softmax_grad.T @ self.queryMatrix
        return self.query_layer.backward(query_grad), self.key_layer.backward(key_grad)
        
class SelfAttentionHead(Layer):
    def __init__(self, sequence_length_in: int, token_length_in: int, token_length_out: int, learning_rate=0.001, name='attention_head'):
        super().__init__()
        self.name='attention_head'
        self.token_length_in = token_length_in
        self.token_length_out = token_length_out
        self.sequence_length = sequence_length_in 

        self.attention_layer = AttentionLayer(sequence_length_in=sequence_length_in, 
                                            token_length_in=token_length_in, 
                                            token_length_out= token_length_out, 
                                            name='attention_layer')
        self.value_layer = FullyConnectedLayer(num_output_nodes=self.token_length_out, num_input_nodes=self.token_length_in, name='value_layer')

    def forward(self, tokens_in: np.array):
        self.values = self.value_layer.forward(tokens_in=tokens_in)
        self.attention = self.attention_layer.forward(tokens_in=tokens_in)
        result = self.attention @ self.values
        return result
    
    def set_value_weights(self, value_weights):
        self.value_layer.set_weights(value_weights)
    def set_key_weights(self, key_weights):
        self.attention_layer.set_key_weights(key_weights)
    def set_query_weights(self, query_weights):
        self.attention_layer.set_query_weights(query_weights)
    def get_value_weights(self):
        return np.copy(self.value_layer.weights)
    def get_key_weights(self):
        return self.attention_layer.get_key_weights()
    def get_query_weights(self):
        return self.attention_layer.get_value_weights()

    def backward(self, upstream_gradient: float):
        super().backward()
        self.value_layer.backward(self.attention.T @ upstream_gradient)
        self.attention_layer.backward(self.values @ upstream_gradient.T)

# Typically dIn is defaulted to sequence_length / h
class MultiHeadSelfAttention:
    # TODO: Create single matrix implementation for multiple heads
    def __init__(self, sequence_length_in: int, token_length_in: int, number_of_heads: int):
        if token_length_in % number_of_heads:
            logging.error(f"number of heads must divide the token length perfectly. \nnum_heads: {number_of_heads}\n token_length: {token_length_in}")
            exit(2)
        self.heads = np.array([], dtype=SelfAttentionHead)
        # Fully connected layer which operates on concatenated output of heads
        self.concatenator = FullyConnectedLayer(token_length_in, token_length_in)
        self.token_length_in = token_length_in
        self.token_length_out = int(token_length_in/number_of_heads)
        for i in range(number_of_heads):
            current_head = SelfAttentionHead(sequence_length_in=sequence_length_in, 
                                            token_length_in=token_length_in, 
                                            token_length_out=self.token_length_out,
                                            name=f'sa_layer_{i}')
            self.heads = np.append(self.heads, current_head)
        

    def forward(self, tokens_in: np.array):
        self.out = np.array([])
        for head in self.heads:
            self.out = np.concatenate(self.out, head.forward(tokens_in), axis=1)
                