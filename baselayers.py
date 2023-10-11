import cupy
import logging
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.name = "layer_unspecified"

    @abstractmethod
    def forward(self):
        """The forward function performs whatever operation is specified by a layer. 
        It also stores the given input for that operation for usage in the backwards pass.
        """
        logging.debug(f"Running forward pass on {self.name}")

    @abstractmethod
    def backward(self):
        """The backwards function performs a the backwards pass using provided upstream gradients.
        This means it updates any weights belonging to the layer by calcuating their gradients,
        and then returns the forward pass's input gradients
        """
        logging.debug(f"Running backward pass on {self.name}")

    # TODO: add shape function


class ReLULayer(Layer):
    """The relu layer rectifies any nonzero values on its input. This class calculates the forward pass for a relu function
    and holds the information necessary to perform the backwards pass of the ReLU layer.
    """

    def __init__(self, name="layer_ReLU"):
        """
        Initializes relu layer with given name
        Args:
            name (str, optional): _description_. Defaults to 'layer_ReLU'.
        """
        super().__init__()
        self.name = name

    def forward(self, data_in):
        """Copies data_in to a new matrix which is stored as self.input. Then, sets any value in self.input less than zero to zero.
        Then, returns a copy of this matrix

        Args:
            data_in (cupy.arange): tensor to be rectified

        Returns:
            cupy.arange: tensor of same shape as data_in with all negative values rectified to 0
        """
        super().forward()
        self.input = cupy.copy(data_in)
        out = cupy.copy(self.input)
        out[out < 0] = 0
        return out

    def backward(self, upstream_gradient):
        """Calculates the gradients for the inputs with respect to an upstream gradient and returns these gradients.
        Uses the most recent forwards pass input in this calculation.

        Args:
            upstream_gradient (cupy.arange): upstream gradients which must be of the same shape as self.input

        Returns:
            cupy.arange: grandients of inputs of last forwards pass (gradients for self.input)
        """
        super().backward()
        if cupy.any(upstream_gradient.shape != self.input.shape):
            logging.log(
                logging.ERROR,
                f"""upstream gradient of {self.name} is not of same shape as input.\n
                input shape: {self.input.shape} upstream_gradient shape: {upstream_gradient.shape}""",
            )
            exit(2)
        return (self.input > 0).astype(cupy.float64) * upstream_gradient

class SoftmaxLayer(Layer):
    """SoftmaxLayer calculates the softmax of an input tensor across any specified axis.
    The default axis is 1, as in a transformer this class is used to softmax each row
    in the attention matrix. The attention matrix is a 2d matrix and therefore to softmax
    each row we take the softmax on axis 1.
    """
    def __init__(self, name="layer_softmax", axis=1, scale=1):
        """_summary_

        Args:
            name (str, optional): Softmax layer name. Defaults to "layer_softmax".
            axis (int, optional): Specified axis to softmax across. Defaults to 1.
            scale (int, optional): Scales each value before softmaxing. Defaults to 1.
        """
        super().__init__()
        name += f"_axis_{axis}"
        name += f"_scaledby_{scale}"
        self.name = name
        self.axis = axis
        self.scale = scale

    def forward(self, data_in):
        super().forward()
        self.I = data_in.shape[0]
        scaled = cupy.divide(data_in, cupy.sqrt(self.scale))
        shifted = cupy.exp(scaled - cupy.max(scaled, axis=self.axis, keepdims=True))
        # Save a copy for the gradient
        softmax = cupy.divide(shifted, cupy.sum(shifted, axis=self.axis, keepdims=True))
        self.output = cupy.copy(softmax)
        return softmax

    def backward(self, upstream_gradient):
        super().backward()
        # Find the jacobian

        # The jacobian is diagonally symmetric
        added_axis = self.output[:, cupy.newaxis]
        num_cols = self.output.shape[1]
        # TODO: investigate
        # jacobian = cupy.repeat(added_axis, num_cols, axis=1)
        jacobian_matrix = cupy.repeat(added_axis, num_cols, axis=1)
        jacobian_matrix *= -1 * self.output.reshape(
            self.output.shape[0], self.output.shape[1], 1
        )
        diagonals = self.output * (1 - self.output)
        jacobian_matrix[:, cupy.arange(num_cols), cupy.arange(num_cols)] = diagonals

        jacobian_matrix *= 1.0 / self.scale
        # Find the gradient
        gradient = cupy.sum(
            upstream_gradient[:, cupy.newaxis, :] * jacobian_matrix, axis=2
        )
        logging.debug(f"Softmax gradient:\n{gradient}")
        return gradient


"""Calculates softmax loss with one hot encoding.
see https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function for more info"""


# TODO: not sure if necessary right now
class SoftmaxClasses(Layer):
    def __init__(self, name="softmax_classifier"):
        super().__init__()

    def get_softmax(self, input_data):
        self.input = input_data
        m = cupy.max(input_data, axis=1, keepdims=True)
        self.softmax = cupy.exp(input_data - m) / cupy.sum(
            cupy.exp(input_data - m), axis=1, keepdims=True
        )
        return self.softmax

    def forward(self, input_data, ground_truth):
        super().forward()
        s = self.get_softmax(input_data)
        s_correct = s[cupy.arange(len(input_data)), ground_truth].reshape(
            (len(input_data), 1)
        )
        return (-1.0 / len(input_data)) * cupy.sum(cupy.log(s_correct))

    def backward(self, ground_truth):
        super().backward()
        y_one_hot = cupy.zeros(self.input.shape)
        y_one_hot[cupy.arange(len(self.input)), ground_truth] = 1
        return (1.0 / len(self.input)) * (self.softmax - y_one_hot)


class FullyConnectedLayer(Layer):
    """
    The fully connected layer is a singular layer of the familiar multi-layer-perceptron model,
    but modified to handle a batch of input data.
    """

    def __init__(
        self,
        num_input_nodes: int,
        num_output_nodes: int,
        name="layer_fc",
        starting_weights_in=None,
        learning_rate=0.001,
    ):
        """
        Input:
        Takes argument num_input_nodes in the input shape and num_output_nodes is the output shape. Name is a string
        which is the name of the layer. starting_weights_in is a matrix which defines the initial weights.


        Modifies:
        Sets self.name to name
        Sets self.weights_shape to (num_input_nodes, num_output_nodes)
        Sets self.weights to starting_weights if specified, or to cupy.ones(self.weights_shape) otherwise.

        Output:
        None
        """
        super().__init__()
        self.name = name
        self.weights = None
        self.set_learning_rate(learning_rate)
        # self.reLU = ReLULayer(name= (self.name + '_ReLU'))
        self.weights_shape = (num_input_nodes, num_output_nodes)
        if not cupy.any(starting_weights_in):
            self.set_weights(starting_weights=cupy.random.rand(*(self.weights_shape)))
        else:
            self.set_weights(starting_weights=starting_weights_in)

    def set_learning_rate(self, learning_rate):
        """
        Input:
        New learning rate to be used

        Modifies:
        Sets self.learning_rate to learning_rate
        """
        self.learning_rate = learning_rate

    def set_weights(self, starting_weights):
        """
        Input:
        Starting weight of shape self.weights_shape. If shape does not match, the program exits and logs an error with shape
        mismatch info.

        Modifies:
        Sets self.weights to starting_weights
        """
        # Initialize weights
        if starting_weights.shape != self.weights_shape:
            logging.error(
                "starting_weights does not match given input and output shape."
            )
            exit(1)
        self.weights = cupy.copy(starting_weights)

    def get_weights(self):
        """
        Output:
        Returns a copy of the weights matrix.
        """
        return cupy.copy(self.weights)

    def forward(self, tokens_in):
        """
        Given an matrix W of shape D x N, forward calculates tokens_in @ W.
        Therefore every column i in w represents the weights for output of the ith token of each sequence.

        Input:
        tokens_in is an I x D matrix, where I is the number of sequences present, and D is the number of tokens in a sequence

        Modifies:
        Sets self.input to a copy of tokens_in for use in backwards pass

        Output:
        I x N matrix, where I is the number of sequences present, and D is the numeber of tokens in a sequence.
        Note: In the query, key and value matrices,N == D, so the input and output are of the same shape.
        """
        super().forward()
        logging.debug(f"input: \n{tokens_in}")
        self.input = cupy.copy(tokens_in)
        if tokens_in.shape[-1] != self.weights.shape[0]:
            logging.error(
                f"""input to {self.name} forward() call does not match weights shape.\n
                        weights shape: {self.weights.shape} input shape: {tokens_in.shape}"""
            )
            exit(1)
        forward_result = tokens_in @ self.weights
        logging.debug(f"output: \n{forward_result}")
        return forward_result
        # return self.reLU.forward(forward_result)

    def backward(self, upstream_gradient, update_weights=True):
        """
        Input:
        Takes in upstream_gradient as a matrix of shape I x N, where I = self.input[0],
        N = forward_result.shape[1] = num_output_nodes


        Modifies:
        Calculates gradient, then adds -learning_rate*gradient to self.weights matrix:
        Self.weights is of shape D x N. The forward pass was calculated by input @ self.weights.
        The gradient with respect to the weight matrix W is self.input.T @ upstream_gradient.
        The gradient with respect to the inputs is upstream_gradient @ self.weights.T.


        Output:
        Calculates the gradient for the current layer and returns the result.
        Resulting gradient should be the same shape as self.weights_shape
        """
        super().backward()
        if not cupy.any(self.input):
            logging.error(
                f"""backward pass called without running forwards pass first!"""
            )
            exit(1)

        logging.debug(f"upstream gradient:\n{upstream_gradient}")
        # reLU_gradient = self.reLU.backward(upstream_gradient)
        weight_gradients = self.input.T @ upstream_gradient
        input_gradients = upstream_gradient @ self.weights.T
        logging.debug(f"local weight_gradients:\n{weight_gradients}")

        if update_weights:
            self.weights -= self.learning_rate * weight_gradients
        return input_gradients


class LayerNorm(Layer):
    def __init__(self, learning_rate=0.01, name="layernorm_layer"):
        self.name = name
        self.eps = 0.001
        self.beta = 0
        self.gamma = 1
        self.learning_rate = 0.01

    def forward(self, input_data):
        super().forward()
        self.input = input_data
        self.N = cupy.size(input_data)
        self.mu = 1.0 / self.N * cupy.sum(input_data)
        self.xmu = input_data - self.mu
        self.xmu_sq = cupy.power(self.xmu, 2)
        self.variance = 1.0 / self.N * cupy.sum(self.xmu_sq)
        self.sqrt_var = cupy.sqrt(self.variance + self.eps)
        self.inverted_variance = 1.0 / self.sqrt_var
        self.normalized = self.xmu * self.inverted_variance
        self.gammax = self.gamma * self.normalized
        self.out = self.gammax + self.beta
        return self.out

    # Trying to calculate the LayerNorm gradient by hand was a horrible idea. so lets do it using a computational graph approach.
    # Adapted from: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    def backward(self, upstream_grad):
        super().backward()
        d_beta = upstream_grad * cupy.sum(self.gammax)
        self.beta -= d_beta * self.learning_rate

        d_gammax = upstream_grad
        d_gamma = cupy.sum(self.normalized * d_gammax)
        self.gamma -= d_gamma * self.learning_rate

        d_normalized = self.gamma * d_gamma

        d_inverted_variance = cupy.sum(self.xmu * d_normalized)
        d_xmu = self.inverted_variance * d_normalized

        d_sqrt_var = -1.0 / (self.sqrt_var**2) * d_inverted_variance

        d_variance = (0.5 * 1.0 / self.sqrt_var) * d_sqrt_var

        d_xmu_sq = 1.0 / self.N * cupy.ones(upstream_grad.shape) * d_variance

        d_xmu = 2.0 * self.xmu * d_xmu_sq

        d_mu = -1.0 * d_xmu

        d_input_data = 1.0 / self.N * d_mu

        return d_input_data
