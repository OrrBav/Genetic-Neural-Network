from buildnet0 import NeuralNetwork, Layer, sigmoid, relu

import numpy as np


def load_test_data(filename):
    with open(filename, 'r') as test_file:
        lines = test_file.readlines()
    data = []
    for line in lines:
        binary_str = line.strip()
        # Convert string to list of ints
        data.append([int(bit) for bit in binary_str])
    return np.array(data)


if __name__ == "__main__":
    best_network = NeuralNetwork()
    loaded_data = np.load("wnet0.npz")
    arr1 = loaded_data['arr1']
    arr2 = loaded_data['arr2']
    arr3 = loaded_data['arr3']
    layers = [arr1, arr2, arr3]
    layer1 = Layer(1, 1, activation=lambda x: relu(x))
    layer1.set_weights(arr1)
    best_network.add_layer(layer1)
    layer2 = Layer(1, 1, activation=lambda x: relu(x))
    layer2.set_weights(arr2)
    best_network.add_layer(layer2)
    layer3 = Layer(1, 1, activation=lambda x: sigmoid(x))
    layer3.set_weights(arr3)
    best_network.add_layer(layer3)

    # load data
    x_test = load_test_data("testnet0.txt")

    # Testing
    test_predictions = best_network.predict(x_test)
    # Write labels to the file
    with open("result0.txt", "w") as file:
        for label in test_predictions:
            file.write(str(label) + "\n")