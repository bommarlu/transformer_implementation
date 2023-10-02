import sys
sys.path.append('/home/bommarlu/projects/attention-is-all-you-need/')
from selfAttention import SelfAttentionHead
import numpy as np
def test_attention_head_forward():
    I = 3
    D = 4
    input_data = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],])
    attention_head = SelfAttentionHead(sequence_length_in = I, token_length_in= D)
    print(attention_head.forward(input_data))


test_attention_head_forward()
