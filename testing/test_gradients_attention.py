import sys
import logging
import copy
sys.path.append('../attention-is-all-you-need/')
from attention import *


def generate_perturbed(shape, delta):
    input_data = cupy.random.rand(*shape)
    return input_data, input_data + delta

def run_forward_perturbed(layer: Layer, input_data, perturbed):
    layer_copy = copy.deepcopy(layer)
    output_data = layer.forward(input_data)
    output_perturbed = layer_copy.forward(perturbed)
    return output_data, output_perturbed

class TestAttentionLayer:
    def test_input_grads(self):
        logging.basicConfig(level=logging.DEBUG)
        delta = 0.001
        def loss_function(output, correct):
            return cupy.sum(cupy.power(output - correct, 2))
        
        # Generate input data
        sequence_length = 2
        token_length = 4
        
        # Calculate an output matrix to optimize to
        random_out = cupy.random.rand(2,2)
        target_out = random_out / cupy.sum(random_out, axis = 1, keepdims=True)

        in_x, in_dx = generate_perturbed([sequence_length,token_length], delta)

        attention = AttentionLayer(sequence_length_in=sequence_length, token_length_in=token_length, token_length_out=token_length)

        out_x, out_dx = run_forward_perturbed(attention, in_x, in_dx)

        gradients = attention.backward(2*(out_x - target_out), update_weights=False)

        predicted_delta = loss_function(out_x, target_out) + cupy.sum(gradients*delta)
        print(predicted_delta, loss_function(out_dx, target_out))
        assert predicted_delta - loss_function(out_dx, target_out) < 0.01
    
    # def test_weight_grads(self):
    #     logging.basicConfig(level=logging.DEBUG)
    #     delta = 0.5
    #     def loss_function(output, correct):
    #         return cupy.sum(cupy.power(output - correct, 2))
        
    #     # Generate input data
    #     sequence_length = 2
    #     token_length = 4
        
    #     # Calculate an output matrix to optimize to
    #     random_out = cupy.random.rand(2,2)
    #     target_out = random_out / cupy.sum(random_out, axis = 1, keepdims=True)

    #     in_x = cupy.random(sequence_length, token_length)

    #     attention = AttentionLayer(sequence_length_in=sequence_length, token_length_in=token_length, token_length_out=token_length)

    #     out_x = attention.forward()

    #     attention.backward(2*(out_x - target_out))
        

    #     predicted_delta = loss_function(out_x, target_out) + cupy.sum(gradients*delta)
    #     assert predicted_delta - loss_function(out_dx, target_out) < 0.01
        
class TestFullyConnectedLayer:
    def test_input_gradient_fc(self):
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
