import selfAttention as sa
import numpy as np

def test_self_attention_head():
    I = 3
    D = 4
    testInput = np.array([[0,0,0,0],[10,10,10,10],[100,100,100,100]])
    saBlock = sa.SelfAttentionHead(sequence_length_in=I, token_length_in=D)
    saBlock.attend(testInput)
    print(saBlock.output)

def test_multi_head_self_attention():
    I = 3
    D = 8
    testInput = np.arange(24).reshape((I, D))
    saBlock = sa.MultiHeadSelfAttention(token_length_in=D, sequence_length_in=I, number_of_heads=2)

if __name__ == "__main__":
    test_self_attention_head()
    test_multi_head_self_attention()
