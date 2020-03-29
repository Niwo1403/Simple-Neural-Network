from numpy import random  # random array for weights
import math  # needed for e in the sigmoid function


class Neuron:
    """
    Represents a single neuron.
    """

    def __init__(self, layer):
        """
        Initializes a neuron with input_count input values.
        :param layer: the neuronal network layer of the neuron
        """
        if layer is not None:
            self.input_weights = 2 * random.random(layer.get_input_count()) - 1
        self.layer = layer
        self.output = None
        self.inputs = None

    @classmethod
    def sigmoid_function(cls, x):
        """
        Evaluates the sigmoid function. f: R -> (-1, 1)
        :param x: the x for f(x)
        :return: f(x) from the sigmoid function
        """
        return 1 / (1 + math.e**-x)

    @classmethod
    def sigmoid_derivation(cls, x):
        """
        Evaluates the derivation of the sigmoid function.
        :param x: the x for f'(x)
        :return: f'(x) from the derivation of the sigmoid function
        """
        sig = Neuron.sigmoid_function(x)
        return sig * (1 - sig)

    def adjust(self, adjustment_factor):
        """
        Adjust the weights of the neuron.
        :param adjustment_factor: the product of the error of the next layer and the learning factor
        :return: the failure of the neuron
        """
        change_factor = Neuron.sigmoid_derivation(self.output) * adjustment_factor
        neuron_failure = sum(map(lambda weight: weight * change_factor, self.input_weights))
        for i in range(len(self.input_weights)):
            self.input_weights[i] -= self.inputs[i] * change_factor
        return neuron_failure

    def think(self):
        """
        Evaluates value of the sigmoid function of the sum off all input values multiplied with their weights.
        :return: the result of the sigmoid function
        """
        self.inputs = list(map(lambda neuron: neuron.get_output(), self.layer.get_inputs()))
        self.output = sum(map(lambda neuron_value, weight: neuron_value * weight,
                              self.inputs, self.input_weights))
        return self.get_output()

    def get_output(self):
        """
        Returns the output of the neuron and generates it before,
        if it's None.
        :return: the  output of the neuron
        """
        if self.output is None:
            self.think()
        return Neuron.sigmoid_function(self.output)


class SimpleInputSupplier(Neuron):
    """
    Used as pseudo neuron for the input layer and as bias.
    """

    def __init__(self, input_value=1):
        """
        Supplies as pseudo neuron the input data for the next layer.
        :param input_value: the value to supply
        """
        super().__init__(None)
        self.value = input_value

    def get_output(self):
        """
        Returns a input value for the first layer.
        :return: the input, which is the 'output' of self
        """
        return self.value

    def think(self):
        """
        It's a dummy class, which can't think, it just returns the constant output.
        :return: the (constant) output
        """
        return self.get_output()

    def adjust(self, adjustment_factor):
        """
        Bias has no weights and doesn't have to be adjusted.
        :param adjustment_factor: irrelevant
        """
        change_factor = Neuron.sigmoid_derivation(self.value) * adjustment_factor
        neuron_failure = self.value * change_factor
        self.value -= change_factor
        return neuron_failure
