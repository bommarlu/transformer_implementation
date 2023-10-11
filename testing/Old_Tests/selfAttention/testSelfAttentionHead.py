import sys
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from attention import AttentionLayer, SelfAttentionHead
import logging
import copy
import numpy as np

def test_sa_backward():
    logging.basicConfig(level=logging.INFO)
    def loss_function(output, correct):
        return np.sum(np.power(output - correct, 2))
    input_data = np.random.rand(2,4)

    #calculate an output matrix
    random_out = np.random.rand(2,4)
    target_out = random_out / np.sum(random_out, axis = 1, keepdims=True)
    target_out *= 10

    attention = SelfAttentionHead(sequence_length_in=2, token_length_in=4, token_length_out=4, learning_rate=0.1)
    loss = None
    while not loss or loss > 0.001:
        logging.debug('============= FORWARD PASS =============')
        forward_pass = attention.forward(input_data)
        loss = loss_function(forward_pass, target_out)

        logging.debug('============= FORWARD PASS DONE =============')
        logging.debug(f'forward_pass result:\n{forward_pass}')
        logging.info(f'loss:{loss}')

        logging.debug('============= BACKWARD PASS =============')
        loss_grad = 2 * (forward_pass - target_out)
        logging.debug(f'output_grad:\n{loss_grad}')
        backward_pass = attention.backward(loss_grad)
    
    print(f'attempt: {attention.forward(input_data)}')
    print(f'target: {target_out}')

def test_multi_sa_backward():
    logging.basicConfig(level=logging.INFO)
    def loss_function(output, correct):
        return np.sum(np.power(output - correct, 2))
    input_data = np.random.rand(2,4)

    #calculate an output matrix
    random_out = np.random.rand(2,4)
    target_out = random_out
    
    attention = SelfAttentionHead(sequence_length_in=2, token_length_in=4, token_length_out=4, learning_rate=0.001)
    attention2 = SelfAttentionHead(sequence_length_in=2, token_length_in=4, token_length_out=4, learning_rate=0.001)
    
    loss = None
    while not loss or loss > 0.3:
        logging.debug('============= FORWARD PASS =============')
        forward_pass = attention2.forward(attention.forward(input_data))
        
        loss = loss_function(forward_pass, target_out)

        logging.debug('============= FORWARD PASS DONE =============')
        logging.debug(f'forward_pass result:\n{forward_pass}')
        logging.info(f'loss:{loss}')

        logging.debug('============= BACKWARD PASS =============')
        loss_grad = 2 * (forward_pass - target_out)
        logging.debug(f'output_grad:\n{loss_grad}')
        backward_pass2 = attention2.backward(loss_grad)
        backward_pass = attention.backward(backward_pass2)
    
    print(f'attempt: {attention2.forward(attention.forward(input_data))}')
    print(f'target: {target_out}')


def test_sa_backward_numerical():
    data_in = np.array([[1,1,1,1],
                        [2,2,2,3]])
    perturbed_up = data_in + 0.1
    perturbed_down = data_in - 0.1


    layer = SelfAttentionHead(sequence_length_in=2, token_length_in=4, token_length_out=4, learning_rate = 0.001)
    original_forward = layer.forward(data_in)
    perturbed_up_forward = copy.deepcopy(layer).forward(perturbed_up)
    perturbed_down_forward = copy.deepcopy(layer).forward(perturbed_down)

    grads = layer.backward(np.ones((2,4)))
    print(grads)
    print((perturbed_up_forward - perturbed_down_forward) / 0.2)

test_sa_backward()
