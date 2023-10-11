import sys
import logging
import copy
sys.path.append('../attention-is-all-you-need/')
from baselayers import *


def generate_perturbed(shape, delta):
    input_data = cupy.random.rand(*shape)
    return input_data, input_data + delta

def run_forward_perturbed(layer: Layer, input_data, perturbed):
    output_data = layer.forward(input_data)
    layercopy = copy.deepcopy(layer)
    output_perturbed = layercopy.forward(perturbed)
    return output_data, output_perturbed

class TestReLULayer:
    def test_gradient_ReLU(self):
        delta = 0.01
        x_in, dx_in = generate_perturbed([100,200], delta)
        relu = ReLULayer()

        x_out, dx_out = run_forward_perturbed(relu, x_in, dx_in)

        gradients = relu.backward(cupy.ones((100, 200)))

        predicted_delta = x_out + (delta * gradients)
        assert not cupy.any(cupy.abs((predicted_delta - dx_out)) > 0.01)
        
class TestFullyConnectedLayer:
    def test_input_gradient_fc(self):
        delta = 100
        num_input = 200
        num_output = 300
        num_samples = 100
        
        def loss(output):
            return cupy.sum(output)
        
        x_in, dx_in = generate_perturbed([num_samples,num_input], delta)
        fc_layer = FullyConnectedLayer(num_input_nodes=num_input, num_output_nodes=num_output)

        x_out, dx_out = run_forward_perturbed(fc_layer, x_in, dx_in)

        gradients = fc_layer.backward(cupy.ones((num_samples, num_output)))

        predicted_delta = loss(x_out) + cupy.sum(gradients) * delta
        assert cupy.abs(predicted_delta - loss(dx_out)) < 0.01

    def test_weight_gradient_fc(self):
        delta = 10
        num_input = 200
        num_output = 300
        num_samples = 100
        
        def loss(output):
            return cupy.sum(output)
        
        x_in, dx_in = generate_perturbed([num_samples,num_input], delta)
        fc_layer = FullyConnectedLayer(num_input_nodes=num_input, num_output_nodes=num_output)

        x_out, dx_out = run_forward_perturbed(fc_layer, x_in, dx_in)

        gradients = fc_layer.backward(cupy.ones((num_samples, num_output)))

        predicted_delta = loss(x_out) + loss(x_in * gradients * delta)
        assert predicted_delta - loss(dx_out) < 0.01
        

class TestLayerNorm:
    def test_gradient_layernorm(self):
        delta = 0.01
        in_x, in_dx = generate_perturbed([100,200], delta)
        norm_layer = LayerNorm()

        out_x, out_dx = run_forward_perturbed(norm_layer, in_x, in_dx)

        gradients = norm_layer.backward(cupy.ones((100, 200)))

        predicted_delta = out_x + (delta * gradients)
        assert not cupy.any(cupy.abs((predicted_delta - out_dx)) > 0.01)

class TestSoftmax:
    
    def test_gradient_softmax(self):
        def loss(output):
            return cupy.sum(cupy.square(output))
         
        delta = 0.5
        in_x = cupy.random.rand(100, 200)
        in_dx = cupy.copy(in_x)
        in_dx[0,0] += delta
        softmax_layer = SoftmaxLayer()

        out_x, out_dx = run_forward_perturbed(softmax_layer, in_x, in_dx)

        gradients = softmax_layer.backward(2*out_x)

        predicted_delta = loss(out_x) + gradients[0,0] * delta
        assert cupy.abs(predicted_delta - loss(out_dx)) < 0.01

