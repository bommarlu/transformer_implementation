import sys
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from selfAttention import AttentionLayer, SelfAttentionHead
import logging
import numpy as np

def test_self_attention_backward():
    logging.basicConfig(level=logging.DEBUG)
    def loss_function(output, correct):
        return np.sum(np.power(output-correct_out, 2))
    input_data = np.random.rand(2,4)
    correct_out = np.random.rand(2,4) * 10
    self_attention = SelfAttentionHead(sequence_length_in=2, token_length_in=4, token_length_out=4, learning_rate=1)
    loss = None
    while not loss or loss > 0.01:
        logging.debug('============= FORWARD PASS =============')
        forward_pass = self_attention.forward(input_data)
        loss = loss_function(forward_pass, correct_out)

        logging.debug(f'forward_pass result:\n{forward_pass}')
        logging.info(f'loss:{loss}')

        logging.debug('============= BACKWARD PASS =============')
        loss_grad = 2 * (forward_pass-correct_out)
        logging.debug(f'output_grad:\n{loss_grad}')
        self_attention.backward(loss_grad)
    print(correct_out)
    print(self_attention.forward(input_data))

test_self_attention_backward()
