from SimpleNeuronalNetwork.NeuralNetworkLayer import NeuralNetworkLayer


class NeuralNetwork:
    """
    Used to generate a neural network.
    """

    def __init__(self, neuron_counts, learning_factor=0.01, bias=1):
        """
        Creates the neural network.
        :param neuron_counts: a list of the neuron count of each layer;
        len(neuron_counts) equals the count of layers
        :param learning_factor: the factor, how much a single result should change the network
        (a huge learning_factor results in a faster change; earlier results will be overridden more for that)
        :param bias: the added base for each network layer (should be 1)
        """
        self.learning_factor = learning_factor
        self.default_bias = bias
        # create layers:
        last_layer = None
        self.layers = []
        for neuron_count in neuron_counts:
            last_layer = NeuralNetworkLayer(neuron_count, last_layer, self.default_bias)
            self.layers.append(last_layer)
        self.layers[-1].to_output_layer()
        self.network_error = 0

    def __iter__(self):
        """
        Iterates over the layers as default.
        :return: an iterator including the layers
        """
        return iter(self.layers)

    @classmethod
    def difference_squared(cls, a, b):
        """
        Evaluates the square of the difference from a and b.
        :param a: the first number (minuend)
        :param b: the second number (subtrahend)
        :return: the square of the difference
        """
        return (a - b)**1#2

    @classmethod
    def loss(cls, a, b):
        """
        Evaluates the difference of a and b and doubles it.
        :param a: the first number (minuend)
        :param b: the second number (subtrahend)
        :return: the double of the difference
        """
        return (a - b) * 2

    def think(self, input_values):
        """
        Starts the neural network with the passed input values.
        :param input_values: the input values for the first layer
        :return: a list of the output of the neural network
        """
        self.layers[0].set_input(input_values)
        for layer in self:
            layer.think()
        return list(map(lambda neuron: neuron.get_output(), self.layers[-1]))

    def train(self, input_values, output_values, iterations=10_000):
        """
        Trains the network with specified input and output values.
        :param input_values: the input values for the network
        :param output_values: the output values for the network
        :param iterations: count of the repetitions of the
        :return: the last error
        """
        for _ in range(iterations):
            network_output = self.think(input_values)
            network_error = sum(map(lambda network_value, desired_value:
                                    NeuralNetwork.loss(network_value, desired_value),
                                    network_output, output_values))
            self.layers[-1].adjust(network_error * self.learning_factor)
        return self.think(input_values)

    def get_error(self):
        """
        Returns the error of the network.
        :return: the error
        """
        return self.network_error

import datetime
start = datetime.datetime.now()
nn = NeuralNetwork([3, 1])  # 3, 1
inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
outputs = [[0],[1],[1],[0]]#[[1, 0], [0, 1], [0, 1], [1, 0]]
for (input_values, output_values) in zip(inputs, outputs):
    nn.train(input_values, output_values)
print("[0, 0, 1] -> 0:  ", nn.think([0, 0, 1]))
print("[1, 1, 1] -> 1:  ", nn.think([1, 1, 1]))
print("[1, 0, 1] -> 1:  ", nn.think([1, 0, 1]))
print("[0, 1, 1] -> 0:  ", nn.think([0, 1, 1]))
print("\n")
print("[1, 0, 0] -> 1:  ", nn.think([1, 0, 0]))
print("[1, 1, 0] -> 1:  ", nn.think([1, 1, 0]))
print("[0, 0, 0] -> 0:  ", nn.think([0, 0, 0]))
print("[0, 1, 0] -> 0:  ", nn.think([1, 1, 0]))
print(datetime.datetime.now(), "\nDauer: ", datetime.datetime.now() - start)
