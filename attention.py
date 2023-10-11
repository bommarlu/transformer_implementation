import logging
import math

import numpy as np

from baselayers import *
from baselayers import FullyConnectedLayer, SoftmaxLayer

# def softmax(x: np.array):
#     raised = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return np.divide(raised, np.sum(raised, axis=1, keepdims=True))

# def scaled_softmax(x: np.array):
#     scaled = np.divide(x, np.sqrt(len(x[0])))
#     raised = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
#     return np.divide(raised, np.sum(raised, axis=1, keepdims=True))

"""Calculates attention matrix of size sequence_length_in * sequence_length_in"""


class AttentionLayer(Layer):
    def __init__(
        self,
        sequence_length_in: int,
        token_head_length: int,
        token_orig_length: int,
        learning_rate=0.001,
        name="attention_layer",
    ):
        super().__init__()
        self.name = name
        self.token_length = token_head_length  # For multiple heads, this is a divisor of token_length in, otherwise 1
        self.token_orig_length = token_orig_length
        self.sequence_length = sequence_length_in
        self.attention_shape = (self.sequence_length, self.sequence_length)

        self.query_layer = FullyConnectedLayer(
            num_output_nodes=self.token_head_length,
            num_input_nodes=self.token_head_length,
            name="query_layer",
            learning_rate=learning_rate,
        )
        self.key_layer = FullyConnectedLayer(
            num_output_nodes=self.token_head_length,
            num_input_nodes=self.token_head_length,
            name="key_layer",
            learning_rate=learning_rate,
        )
        self.attention = None
        self.softmax_layer = SoftmaxLayer(
            name="attention_softmax_layer", axis=1, scale=np.sqrt(token_orig_length)
        )

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
        logging.debug(f"Query Weights:\n{self.query_layer.get_weights()}")
        logging.debug(f"Key Weights:\n{self.key_layer.get_weights()}")

        query_key_expected_shape = (self.sequence_length, self.token_head_length)

        self.queryMatrix = self.query_layer.forward(
            tokens_head_in=tokens_in
        )  # Should be an I * D/H matrix
        assert (
            self.queryMatrix.shape == query_key_expected_shape
        ), f"Query matrix shape is incorrect. Expected {query_key_expected_shape}, got {self.queryMatrix.shape}"

        self.keyMatrix = self.key_layer.forward(
            tokens_head_in=tokens_in
        )  # Should be an I * D/H matrix
        assert (
            self.keyMatrix.shape == query_key_expected_shape
        ), f"Key matrix shape is incorrect. Expected {query_key_expected_shape}, got {self.keyMatrix.shape}"

        softmax_expected_shape = (self.sequence_length, self.sequence_length)
        pre_softmax = self.queryMatrix @ self.keyMatrix.T  # should be an I * I matrix
        assert (
            pre_softmax.shape == softmax_expected_shape
        ), f"Pre softmax shape is incorrect. Expected {softmax_expected_shape}, got {pre_softmax.shape}"
        logging.debug(f"Attention matrix, pre softmax:\n{pre_softmax}")

        self.attention = self.softmax_layer.forward(
            data_in=pre_softmax
        )  # should be an I * I matrix
        assert (
            self.attention.shape == softmax_expected_shape
        ), f"Attention matrix shape is incorrect. Expected {softmax_expected_shape}, got {self.attention.shape}"
        logging.debug(f"Attention matrix:\n{self.attention}")

        return np.copy(self.attention)

    def backward(self, upstream_grad, update_weights=True):
        super().backward()
        self.softmax_grad = self.softmax_layer.backward(upstream_grad)
        self.query_grad = self.softmax_grad @ self.keyMatrix
        self.key_grad = self.softmax_grad.T @ self.queryMatrix
        return self.query_layer.backward(
            self.query_grad, update_weights
        ) + self.key_layer.backward(self.key_grad, update_weights)


class SelfAttentionHead(Layer):
    def __init__(
        self,
        sequence_length_in: int,
        token_head_length: int,
        token_orig_length: int,
        learning_rate=0.001,
        name="attention_head",
    ):
        super().__ihead_nit__()
        self.name = "attention_head"
        self.token_head_length = token_head_length  # D/H
        self.token_orig_length = token_orig_length  # D
        self.sequence_length = sequence_length_in  # I

        self.attention_layer = AttentionLayer(
            sequence_length_in=sequence_length_in,
            token_head_length=token_head_length,
            token_orig_length=token_orig_length,
            learning_rate=learning_rate,
            name="attention_layer",
        )
        self.value_layer = FullyConnectedLayer(
            num_output_nodes=self.token_head_length,
            num_input_nodes=self.token_head_length,
            learning_rate=learning_rate,
            name="value_layer",
        )

    def forward(self, tokens_in: np.array):
        super().forward()
        self.values = self.value_layer.forward(tokens_in=tokens_in)
        self.attention = self.attention_layer.forward(tokens_in=tokens_in)
        result = self.attention @ self.values
        return result

    def backward(self, upstream_gradient: float):
        super().backward()
        value_grad = self.value_layer.backward(self.attention.T @ upstream_gradient)
        attention_grad = self.attention_layer.backward(
            self.values @ upstream_gradient.T
        )
        return value_grad + attention_grad

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


# Typically dIn is defaulted to sequence_length / h
class MultiHeadSelfAttention(Layer):
    # TODO: Create single matrix implementation for multiple heads
    def __init__(
        self,
        sequence_length_in: int,
        token_length_in: int,
        number_of_heads: int,
        learning_rate=0.01,
        name="multi_head_layer",
    ):
        if token_length_in % number_of_heads:
            logging.error(
                f"number of heads must divide the token length perfectly. \nnum_heads: {number_of_heads}\n token_length: {token_length_in}"
            )
            exit(2)
        self.heads = []
        self.name = name
        # Fully connected layer which operates on concatenated output of heads

        # concatenates the SelfAttentionHEads into a single output
        self.concatenator = FullyConnectedLayer(
            token_length_in, token_length_in, learning_rate=learning_rate
        )

        # The full dimensionality of the embedding length for each token
        self.token_length_in = token_length_in

        # the dimensionality of the embedding length for each token in each head
        self.token_head_length = int(token_length_in / number_of_heads)

        for i in range(number_of_heads):
            # Creates a new head with same sequence length and divided embedding length
            current_head = SelfAttentionHead(
                sequence_length_in=sequence_length_in,
                token_head_length=self.token_head_length,
                token_orig_length=self.token_length_in,
                learning_rate=learning_rate,
                name=f"{self.name}_sa_layer_{i}",
            )
            self.heads.append(current_head)

    def forward(self, tokens_in: np.array):
        super().forward()
        outputs = []

        tokens_split_for_head = np.array_split(tokens_in, len(self.heads), axis=1)

        for head, token_split in zip(self.heads, tokens_split_for_head):
            outputs.append(head.forward(token_split))
        return self.concatenator.forward(np.concatenate(outputs, axis=1))

    def backward(self, upstream_grad):
        super().backward()
        backwards_grad = self.concatenator.backward(upstream_gradient=upstream_grad)
        sections = np.split(backwards_grad, len(self.heads), axis=1)
        input_grads = []
        for idx, head in enumerate(self.heads):
            head.backward(sections[idx])
