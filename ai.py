from SimpleNeuralNetwork.NeuralNetwork import NeuralNetwork


# First example
# for the bits x1, x2, x3 the boolean expression for the task is: x1
print("boolean expression: x1")

# data to learn:
train_inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
train_outputs = [[1, 0], [0, 1], [0, 1], [1, 0]]
test_inputs = [[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]]
test_outputs = [[0, 1], [0, 1], [1, 0], [1, 0]]
# create and train network:
nn = NeuralNetwork([3, 2])
nn.train(train_inputs, train_outputs)
# print results:
nn.print_results(train_inputs, train_outputs, "Known data:")
nn.print_results(test_inputs, test_outputs, "Unknown data:")
print("\n-------------------------\n")


# Second example
# for the bits x1, x2, x3 the boolean expression for the task is: x1 && x2 || x2 && x3 || x1 && x3
print("boolean expression: x1 && x2 || x2 && x3 || x1 && x3")

# data to learn:
train_inputs = [[1, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]]
train_outputs = [[1, 0], [1, 0], [0, 1], [0, 1]]
test_inputs = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1]]
test_outputs = [[1, 0], [1, 0], [0, 1], [0, 1]]
# create and train network:
nn = NeuralNetwork([3, 2])
nn.train(train_inputs, train_outputs)
# print results:
nn.print_results(train_inputs, train_outputs, "Known data:")
nn.print_results(test_inputs, test_outputs, "Unknown data:")
