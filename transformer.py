from attention import MultiHeadSelfAttention
from fullyconnected import FullyConnected2Layer
from baselayers import LayerNorm, Layer

class transformer(Layer):
    def __init__(self, sequence_length, learning_rate=0.01):
        super().__init__()
        self.input = None
        self.layernorm1 = LayerNorm(name='pre_sa_layernorm_layer', learning_rate=learning_rate)
        self.multi_head = MultiHeadSelfAttention(sequence_length_in=sequence_length, learning_rate=learning_rate)
        self.layernorm2 = LayerNorm(name='pre_fc_layernorm_layer', learning_rate=learning_rate)
        self.fc = FullyConnected2Layer()


    def forward(self, sequence_in):
        super().forward()
        residual = sequence_in

        layernorm1_out = self.layernorm1.forward(sequence_in)
        multi_head_out = self.multi_head(layernorm1_out)

        # Add sequence in for residual effect
        layernorm2_out = self.layernorm2.forward(multi_head_out + residual)
        residual = multi_head_out + residual

        fc_out = self.fc.forward(layernorm2_out)
        return fc_out + residual
