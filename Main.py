import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Genetic Algorithm parameters
population_size = 50
mutation_rate = 0.1
num_generations = 20
elite_size = 0.5

# Neural Network parameters
input_size = 16
hidden_size_1 = 8
hidden_size_2 = 8
output_size = 1

# Data preparation
def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 2d array where each row is a list
    data = []
    labels = []
    for line in lines:
        binary_str, label = line.strip().split()
        data.append([int(bit) for bit in binary_str])  # Convert string to list of ints
        labels.append(int(label))  # Convert label to int

    return np.array(data), np.array(labels)


# get the data and labels from chosen txt file
data, labels = load_data("nn0.txt")
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)


# Neural Network Definition
def create_neural_network():
    model = NeuralNetwork()
    # TODO: add more hidden layers
    model.add_layer(Layer(input_size, hidden_size_1))
    model.add_layer(Layer(hidden_size_1, hidden_size_2))
    # activation layer
    model.add_layer(Layer(hidden_size_2, output_size, 1))
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
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.nn_list = None

    def evolve(self, x_train, y_train, num_generations):
        population = []
        for _ in range(self.population_size):
            network = create_neural_network()
            population.append(network)

        for generation in range(num_generations):
            print(f"Generation {generation+1}/{num_generations}")

            # Evaluation
            fitness_scores = []
            for network in population:
                fitness = evaluate_fitness(network, x_train, y_train)
                fitness_scores.append(fitness)

            print(f"Generation {generation+1} best score is: {max(fitness_scores)}")
            # Selection
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_population = [population[i] for i in sorted_indices[:int(self.population_size * elite_size)]]
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
                offspring.mutate(self.mutation_rate)

            # Combine selected and offspring populations
            population = selected_population + offspring_population


        # Select the best individual from the final population
        fitness_scores = [evaluate_fitness(network, x_train, y_train) for network in population]
        best_individual = population[np.argmax(fitness_scores)]
        return best_individual

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

    def mutate(self, mutation_rate):
        """
        the mutation process randomly selects a subset of weights in each layer based on the mutation rate.
        For the selected weights, a random value is added to introduce variation. This helps in exploring different
        regions of the solution space during the genetic algorithm optimization process.
        :param mutation_rate:
        :return:
        """
        for layer in self.layers:
            mask = np.random.rand(*layer.weights.shape) < mutation_rate
            layer.weights[mask] += np.random.randn(*layer.weights.shape)[mask]


# network = create_neural_network()
# fitness_scores = []
# for _ in range(20):
#     fitness = evaluate_fitness(network, x_train, y_train)
#     fitness_scores.append(fitness)
#
# print(fitness_scores)

# Main
genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate)
best_network = genetic_algorithm.evolve(x_train, y_train, num_generations)

# Testing
predictions = best_network.predict(x_test)
accuracy = compute_accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy}")
