import sys
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from selfAttention import AttentionLayer, SelfAttentionHead
import logging
import numpy as np

def test_sa_bacward():
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


def test_sa_backward_numerical():
    data_in = np.array([[1,1,1,1],
                        [2,2,2,3]])
    perturbed = data_in + 0.01

    layer = SelfAttentionHead(sequence_length_in=2, token_length_in=4, token_length_out=4, learning_rate = 0.001)
    original_forward = layer.forward(data_in)
    perturbed_forward = layer.forward(perturbed)

    grads = layer.backward(original_forward)
    print((1 + grads) * original_forward)
    print(perturbed_forward)

test_sa_backward_numerical()
