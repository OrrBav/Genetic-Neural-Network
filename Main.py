import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Genetic Algorithm parameters
population_size = 50
mutation_rate = 0.1
num_generations = 20

# Neural Network parameters
input_size = 16
hidden_size = 8
output_size = 1

# Data preparation
data = np.array([
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
])  # Example dataset
labels = np.array([1, 0])  # Example labels

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Encoding
x_train_encoded = x_train.reshape((-1, input_size))
x_test_encoded = x_test.reshape((-1, input_size))

# Neural Network Definition
def create_neural_network():
    model = NeuralNetwork()
    model.add_layer(Layer(input_size, hidden_size))
    model.add_layer(Layer(hidden_size, output_size))
    return model

# Fitness Function
def evaluate_fitness(network, x_train, y_train):
    predictions = network.predict(x_train)
    return accuracy_score(y_train, predictions)

# Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

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

            # Selection
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_population = [population[i] for i in sorted_indices[:self.population_size // 2]]

            # Crossover
            offspring_population = []
            for _ in range(self.population_size - len(selected_population)):
                parent1 = np.random.choice(selected_population)
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
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)

    def forward(self, inputs):
        return np.dot(inputs, self.weights)

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs.flatten()

    def crossover(self, other_network):
        new_network = create_neural_network()
        for i in range(len(self.layers)):
            if np.random.rand() > 0.5:
                new_network.layers[i].weights = self.layers[i].weights
            else:
                new_network.layers[i].weights = other_network.layers[i].weights
        return new_network

    def mutate(self, mutation_rate):
        for layer in self.layers:
            mask = np.random.rand(*layer.weights.shape) < mutation_rate
            layer.weights[mask] += np.random.randn(*layer.weights.shape)[mask]

# Main
genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate)
best_network = genetic_algorithm.evolve(x_train_encoded, y_train, num_generations)

# Testing
predictions = best_network.predict(x_test_encoded)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy}")
