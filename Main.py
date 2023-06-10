import numpy as np

# Genetic Algorithm parameters
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
GENERATIONS = 20
ELITE_SIZE = 0.5

# Neural Network parameters
INPUT_SIZE = 16
HIDDEN_SIZE_1 = 8
HIDDEN_SIZE_2 = 8
OUTPUT_SIZE = 1


def load_data(filename):
    """
        Data preparation. Load the binary strings data and their corresponding labels from the input text file.
        Returns:
        'data'- numpy 2D array where each row is a binary string from a file line, converted to a list of integers.
        'labels'- numpy 1D array of the labels corresponding to each binary string, converted to integers.
        """
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    labels = []
    for line in lines:
        binary_str, label = line.strip().split()
        data.append([int(bit) for bit in binary_str])  # Convert string to list of ints
        labels.append(int(label))  # Convert label to int

    # data is a 2d array where each row is a list
    return np.array(data), np.array(labels)


def split_train_test(data, labels, test_size=0.2):
    """
        Split data and labels into a train set and a test set.
        the test_size is a fraction of the data to be used as test data.
        Returns x_train (train data), x_test (test data), y_train (train labels), y_test (test labels)
    """
    # Calculate the number of test samples
    num_test_samples = int(len(data) * test_size)
    # Generate random indices for the test set
    test_indices = np.random.choice(len(data), size=num_test_samples, replace=False)
    # Generate train indices as the complement of the test indices
    train_indices = np.setdiff1d(np.arange(len(data)), test_indices)
    # Split the data and labels
    x_train = data[train_indices]
    x_test = data[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    return x_train, x_test, y_train, y_test


# get the data and labels from chosen txt file
data, labels = load_data("nn0_test_file.txt")
# Split the data into train and test sets
x_train, x_test, y_train, y_test = split_train_test(data, labels, test_size=0.2)


# Neural Network Definition
def create_neural_network():
    model = NeuralNetwork()
    # TODO: add more hidden layers
    model.add_layer(Layer(INPUT_SIZE, HIDDEN_SIZE_1))
    model.add_layer(Layer(HIDDEN_SIZE_1, HIDDEN_SIZE_2))
    # activation layer
    model.add_layer(Layer(HIDDEN_SIZE_2, OUTPUT_SIZE, 1))
    return model


def compute_accuracy_score(y_train, predictions):
    num_samples = len(y_train)
    correct_predictions = 0

    for true_label, predicted_label in zip(y_train, predictions):
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / num_samples
    return accuracy


# Fitness Function
def evaluate_fitness(network, x_train, y_train):
    predictions = network.predict(x_train)
    # todo: implement our own accuracy_score
    # return accuracy_score(y_train, predictions)
    return compute_accuracy_score(y_train, predictions)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Genetic Algorithm
class GeneticAlgorithm:

    def __init__(self):
        self.population_size = POPULATION_SIZE

    def evolve(self, x_train, y_train):
        population = []
        for _ in range(self.population_size):
            network = create_neural_network()
            population.append(network)

        for generation in range(GENERATIONS):
            print(f"Generation {generation+1}/{GENERATIONS}")

            # Evaluation
            fitness_scores = []
            for network in population:
                fitness = evaluate_fitness(network, x_train, y_train)
                fitness_scores.append(fitness)

            print(f"Generation {generation+1} best score is: {max(fitness_scores)}")
            # Selection
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_population = [population[i] for i in sorted_indices[:int(self.population_size * ELITE_SIZE)]]
            remaining_population = list(set(population) - set(selected_population))

            # Crossover
            offspring_population = []
            for _ in range(self.population_size - len(selected_population)):
                parent1 = np.random.choice(remaining_population)
                parent2 = np.random.choice(selected_population)
                offspring = parent1.crossover(parent2)
                offspring_population.append(offspring)

            # Mutation
            for offspring in offspring_population:
                offspring.mutate()

            # Combine selected and offspring populations
            population = selected_population + offspring_population


        # Select the best individual from the final population
        fitness_scores = [evaluate_fitness(network, x_train, y_train) for network in population]
        best_individual = population[np.argmax(fitness_scores)]
        return best_individual

    def lemarkian_evolution(self):
        pass


# Neural Network Implementation
class Layer:
    def __init__(self, input_size, output_size, activation=0):
        # todo: change to a different distribution?
        self.weights = np.random.randn(input_size, output_size)
        self.activation = activation

    def forward(self, inputs):
        output = np.dot(inputs, self.weights)
        if self.activation:
            output = sigmoid(output)
        return output


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        binary_predictions = (outputs > 0.5).astype(int)
        return binary_predictions.flatten()

    def crossover(self, other_network):
        new_network = create_neural_network()
        for i in range(len(self.layers)):
            if np.random.rand() > 0.5:
                new_network.layers[i].weights = np.copy(self.layers[i].weights)
            else:
                new_network.layers[i].weights = np.copy(other_network.layers[i].weights)
        return new_network

    def mutate(self):
        """
        the mutation process randomly selects a subset of weights in each layer based on the mutation rate.
        For the selected weights, a random value is added to introduce variation.
        This helps in exploring different regions of the solution space during the genetic algorithm optimization process
        """
        for layer in self.layers:
            mask = np.random.rand(*layer.weights.shape) < MUTATION_RATE
            layer.weights[mask] += np.random.randn(*layer.weights.shape)[mask]



# Main
genetic_algorithm = GeneticAlgorithm()
best_network = genetic_algorithm.evolve(x_train, y_train)

# Testing
predictions = best_network.predict(x_test)
accuracy = compute_accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy}")
