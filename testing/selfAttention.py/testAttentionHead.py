import sys
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from selfAttention import AttentionLayer, SelfAttentionHead
import logging
import numpy as np



# def test_attention_backward():
#     def loss_function(output):
#         return np.sum(np.power(output, 2))
#     input_data = np.random.rand(2,4)
#     attention = AttentionLayer(sequence_length_in=2, token_length_in=4)
#     loss = None
#     while not loss or loss > 1:
#         logging.debug('============= FORWARD PASS =============')
#         forward_pass = attention.forward(input_data)
#         loss = loss_function(forward_pass)

#         logging.debug(f'forward_pass result:\n{forward_pass}')
#         logging.debug(f'loss:{loss}')

#         logging.debug('============= BACKWARD PASS =============')
#         loss_grad = 2 * forward_pass
#         logging.debug(f'output_grad:\n{loss_grad}')
#         backward_pass = attention.backward(loss_grad)
#         logging.debug(f'backward_pass query:\n{backward_pass[0]}')
#         logging.debug(f'backward_pass keys:\n{backward_pass[1]}' )


def test_attention_backward():
    logging.basicConfig(level=logging.INFO)
    def loss_function(output):
        return np.sum(np.power(output, 2))
    input_data = np.random.rand(2,4)
    attention = AttentionLayer(sequence_length_in=2, token_length_in=4, learning_rate=0.001)
    loss = None
    while not loss or loss > 1.001:
        logging.debug('============= FORWARD PASS =============')
        forward_pass = attention.forward(input_data)
        loss = loss_function(forward_pass)

        logging.debug(f'forward_pass result:\n{forward_pass}')
        logging.info(f'loss:{loss}')

        logging.debug('============= BACKWARD PASS =============')
        loss_grad = 2 * forward_pass
        logging.debug(f'output_grad:\n{loss_grad}')
        backward_pass = attention.backward(loss_grad)
        logging.debug(f'backward_pass query:\n{backward_pass[0]}')
        logging.debug(f'backward_pass keys:\n{backward_pass[1]}' )
    attention.forward(input_data)

test_attention_backward()
