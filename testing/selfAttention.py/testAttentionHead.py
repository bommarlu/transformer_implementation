import sys
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from selfAttention import AttentionLayer, SelfAttentionHead
from logger import logger 
import numpy as np



def test_attention_backward_trivial():
    def loss_function(output):
        return np.sum(np.power(output, 2))
    input_data = np.array([[1, 1, 1, 1],
                            [1, 1,1, 1]])
    attention = AttentionLayer(sequence_length_in=2, token_length_in=4)
    loss = None
    while not loss or loss > 0.5:
        logger.info('============= FORWARD PASS =============')
        forward_pass = attention.forward(input_data)
        loss = loss_function(forward_pass)

        logger.info(f'forward_pass result:\n{forward_pass}')
        logger.info(f'loss:{loss}')

        logger.info('============= BACKWARD PASS =============')
        loss_grad = 2 * forward_pass
        logger.info(f'output_grad:\n{loss_grad}')
        backward_pass = attention.backward(loss_grad)
        logger.info(f'backward_pass query:\n{backward_pass[0]}')
        logger.info(f'backward_pass keys:\n{backward_pass[1]}' )



def test_attention_backward():
    def loss_function(output):
        return np.linalg.norm(output, ord=2)
    input_data = np.array([[1, 2, 3, 4],
                           [5, 5, 5, 5]])
    attention = AttentionLayer(sequence_length_in=2, token_length_in=4)
    loss = None
    while not loss or loss > 1:
        logger.info('============= FORWARD PASS =============')
        forward_pass = attention.forward(input_data)
        loss = loss_function(forward_pass)

        logger.info(f'forward_pass result:\n{forward_pass}')
        logger.info(f'loss:{loss}')

        logger.info('============= BACKWARD PASS =============')
        loss_grad = 2 * forward_pass
        logger.info(f'output_grad:\n{loss_grad}')
        backward_pass = attention.backward(loss_grad)
        logger.info(f'backward_pass query:\n{backward_pass[0]}')
        logger.info(f'backward_pass keys:\n{backward_pass[1]}' )


test_attention_backward()
