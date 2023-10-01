import numpy as np

def relu(x: np.array):
        x[x < 0] = 0

def softmax(x: np.array):
    raised = np.exp(x - np.max(x, axis=1, keepdims=True))
    return np.divide(raised, np.sum(raised, axis=1, keepdims=True))

def scaled_softmax(x: np.array):
    scaled = np.divide(x, np.sqrt(len(x[0])))
    raised = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return np.divide(raised, np.sum(raised, axis=1, keepdims=True))

class SelfAttentionHead:
    def __init__(self, sequence_length_in: int, token_length_in: int,):
        self.token_length = token_length_in
        self.sequence_length = sequence_length_in 

        
        self.val_weights = np.random.rand(token_length_in, token_length_in)
        self.query_weights = np.random.rand(token_length_in, token_length_in)
        self.key_weights = np.random.rand(token_length_in, token_length_in)


        self.queries = np.zeros((token_length_in, token_length_in))
        self.keys = np.zeros((token_length_in, token_length_in))
        self.values = np.zeros((token_length_in, token_length_in))
        self.attention_weights = np.zeros((sequence_length_in, sequence_length_in))
        self.output =  np.zeros((sequence_length_in, token_length_in))

    def attend(self, inputToken: np.array):
        assert(inputToken.shape == (self.sequence_length, self.token_length))
        self.queries = inputToken @ self.query_weights
        self.keys = inputToken @ self.key_weights
        self.values = inputToken @ self.val_weights
        self.attention_weights = scaled_softmax(self.queries @ self.keys.T)
        self.output = self.attention_weights @ self.values

# Typically dIn is defaulted to sequence_length / h
class MultiHeadSelfAttention:
    def __init__(self, sequence_length_in: int, token_length_in: int, number_of_heads: int):
        self.heads = np.array([], dtype=SelfAttentionHead)
        for _ in range(number_of_heads):
            self.heads = np.append(self.heads, SelfAttentionHead(sequence_length_in, token_length_in))
        print(self.heads)
        
