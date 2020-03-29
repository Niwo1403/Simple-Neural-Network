from SimpleNeuronalNetwork.Neuron import Neuron
from SimpleNeuronalNetwork.Neuron import SimpleInputSupplier


class NeuralNetworkLayer:
    """
    A layer of a neural network, including the neurons.
    """

    def __init__(self, neuron_count, previous=None, bias=1):
        """
        Creates a layer of a neural network.
        :param neuron_count: count of neurons in a layer
        :param previous: the object of the previous network layer, or None if it's the input layer
        :param bias: the bias for the neural network
        """
        self.neuron_count = neuron_count + 1  # for bias
        self.prev_layer = previous
        self.default_bias = bias
        self.neurons = [Neuron(self) for _ in range(neuron_count)]
        self.neurons.append(SimpleInputSupplier(bias))
        self.expected_values = None
        self.is_output_layer = False

    def __iter__(self):
        """
        Creates a iterator for the neurons of the layer.
        :return: the iterator for the neurons
        """
        return iter(self.neurons)

    @classmethod
    def create_input_layer(cls, neuron_count):
        """
        Creates the input layer for a neural network.
        :param neuron_count: count of input neurons
        :return: the created neuronal layer
        """
        return NeuralNetworkLayer(neuron_count, None, 1)

    def adjust(self, output_error):
        """
        Adjusts the weights of the neurons in this layer of the neural network.
        :param output_error: the error of the layer multiplied by the learning rate
        """
        layer_error = 0
        for neuron in self:
            layer_error += neuron.adjust(output_error)
        self.prev_layer.adjust(layer_error)

    def set_input(self, input_values, bias=None):
        """
        Used in the input layer to set the inputs.
        :param input_values: a list of the input values
        :param bias: the bias for the input, if not passed the default bias from the constructor is used
        """
        if bias is None:
            input_values.append(self.default_bias)
        else:
            input_values.append(bias)
        self.prev_layer = NeuralNetworkLayer.SimpleNetworkSupplier(input_values)

    def think(self):
        """
        Calls the think method for all neurons of the layer.
        """
        if self.prev_layer is None:
            print("Error, no input data set.")
            return None
        for neuron in self:
            neuron.think()

    def get_input_count(self):
        """
        Evaluates the count of inputs for this layer.
        Equals the count of neurons of the previous layer, or inputs in input layer.
        :return: the count of inputs for neurons in this layer
        """
        if self.prev_layer is None:
            return self.get_neuron_count()
        else:
            return self.prev_layer.get_neuron_count()

    def get_neuron_count(self):
        """
        Returns the count of neurons of this layer.
        :return: the count of neurons
        """
        return self.neuron_count

    def get_inputs(self):
        """
        Returns the inputs for the neurons.
        :return: the list of inputs
        """
        return iter(self.prev_layer)

    def to_output_layer(self):
        """
        Sets the layer as output layer and removes the bias.
        """
        if not self.is_output_layer:
            self.is_output_layer = True
            self.neurons = self.neurons[:-1]

    class SimpleNetworkSupplier:
        """
        Used as pseudo layer for the input layer.
        """

        def __init__(self, input_values):
            """
            Creates a supplier for all input values as pseudo layer.
            :param input_values: a list of input values pseudo neurons should be created for
            """
            self.input_neurons = [SimpleInputSupplier(value) for value in input_values]

        def __iter__(self):
            """
            Returns a iterator of pseudo neurons for the input layer.
            :return: the iterator over the neurons
            """
            return iter(self.input_neurons)

        def get_neuron_count(self):
            """
            Returns the count of pseudo neurons for the input.
            :return: the count of neurons
            """
            return len(self.input_neurons)

        def adjust(self, adjustment_factor):
            """
            Doesn't include neurons and so adjust call has no effect.
            :param adjustment_factor: irrelevant
            """
            pass
