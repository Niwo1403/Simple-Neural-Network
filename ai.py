from SimpleNeuronalNetwork.NeuralNetwork import NeuralNetwork

# data to learn:
train_inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
train_outputs = [[1, 0], [0, 1], [0, 1], [1, 0]]
test_inputs = [[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]]
test_outputs = [[0, 1], [0, 1], [1, 0], [1, 0]]

# create and train network:
nn = NeuralNetwork([3, 3, 2])
nn.train(train_inputs, train_outputs)

# print results:
nn.print_results(train_inputs, train_outputs, "Known data:")
nn.print_results(test_inputs, test_outputs, "Unknown data:")
