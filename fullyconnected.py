import cupy
import logging
import math
from baselayers import SoftmaxClasses, FullyConnectedLayer, ReLULayer
from baselayers import *


class FullyConnected2Layer(Layer):
    def __init__(self, input_shape, num_output_classes, hidden_layer_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size + 1

        self.fc_1 = FullyConnectedLayer(input_shape + 1, hidden_layer_size)
        self.relu_1 = ReLULayer()

        self.fc_2 = FullyConnectedLayer(self.hidden_layer_size, num_output_classes)
        self.relu_2 = ReLULayer()

        self.softmax_classes = SoftmaxClasses()
    
    def forward(self, data_in):
        super().forward()
        self.input = cupy.append(data_in, cupy.ones(data_in.shape[0]), axis=1)
        out_fc1 = cupy.append(self.fc_1.forward(self.input), cupy.ones(self.hidden_layer_size + 1), axis=1)
        self.relu_1.forward(out_fc1)
        out_fc2 = self.fc_2.forward(data_in)
        relu2 = self.relu_2.forward(out_fc2)
        self.output = self.softmax_classes.get_softmax(relu2)

    def backward(self, ground_truth):
        super().backward()
        loss = self.softmax_classes.backward(ground_truth)
        backward_relu_2 = self.relu_2.backward(loss)
        backward_fc_2 = self.fc_2.backward(backward_relu_2)
        
        backward_relu_1 = self.relu_1.backward(backward_fc_2)
        self.fc_1.backward(backward_relu_1)
