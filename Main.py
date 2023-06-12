import numpy as np
import copy
from statistics import mean

from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning)

# Genetic Algorithm parameters
POPULATION_SIZE = 250
MUTATION_RATE = 0.3
GENERATIONS = 200
ELITE_SIZE = 0.1
OFFSPRING_UNTOUCHED = 0.3
STUCK_THRESHOLD = 25
LAMARCKIAN_MUTATIONS = 3

# Neural Network parameters
INPUT_SIZE = 16
HIDDEN_SIZE_1 = 16
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
data, labels = load_data("nn0.txt")
# Split the data into train and test sets
x_train, x_test, y_train, y_test = split_train_test(data, labels, test_size=0.2)


def create_neural_network():
    """
       creates and initialize a neural network with three layers: two hidden layers and one output layer.
       The sizes of the input layer, hidden layers, and output layer are determined by global variables.
       The output layer includes an activation function.
       """
    model = NeuralNetwork()
    # TODO: add more hidden layers
    model.add_layer(Layer(INPUT_SIZE, HIDDEN_SIZE_1))
    model.add_layer(Layer(HIDDEN_SIZE_1, OUTPUT_SIZE, 1))
    # activation layer
    # model.add_layer(Layer(HIDDEN_SIZE_2, OUTPUT_SIZE, 1))
    return model


def compute_accuracy_score(y_train, predictions):
    """
        Calculates the accuracy score of the predictions made by the network.
        If a prediction matches its true label, it increments the count of correct predictions.
        The accuracy score is the correct predictions divided by the total predictions.
    """
    num_samples = len(y_train)
    correct_predictions = 0

    # If the prediction is correct, increment the count of correct predictions
    for true_label, predicted_label in zip(y_train, predictions):
        if true_label == predicted_label:
            correct_predictions += 1

    # Compute accuracy as the ratio of correct predictions to total number of samples
    accuracy = correct_predictions / num_samples
    return accuracy

# attempt for a different fitness function - seems to work worse
# from sklearn.metrics import precision_score
# def evaluate_fitness(network, x_train, y_train):
#     predictions = network.predict(x_train)
#     return precision_score(y_train, predictions)


def evaluate_fitness(network, x_train, y_train):
    """
        The fitness function evaluates how well the neural network performs on the training data.
        It uses the accuracy of the network's predictions as the fitness score.
    """
    predictions = network.predict(x_train)
    return compute_accuracy_score(y_train, predictions)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GeneticAlgorithm:
    """
        This class represents a genetic algorithm for optimizing the structure and parameters of a neural network.
        The algorithm starts by creating an initial population of random neural networks.
        Then, over a series of generations, it evaluates the performance (fitness) of each network,
        selects the best performers, and breeds a new generation of networks through crossover and mutation.
        At the end of the generations, the algorithm selects the best overall network and returns it.
    """

    def __init__(self):
        self.population_size = POPULATION_SIZE

    def evolve(self, x_train, y_train):
        # Creating an initial population of neural networks
        population = []
        for _ in range(self.population_size):
            network = create_neural_network()
            population.append(network)

        best_fitness_so_far = 0
        gen_stuck_count = 0
        for generation in range(GENERATIONS):
            print(f"Generation {generation+1}/{GENERATIONS}")

            # Evaluating the fitness of each network in the current population
            fitness_scores = []
            for network in population:
                fitness = evaluate_fitness(network, x_train, y_train)
                fitness_scores.append(round(fitness, 5))

            curr_gen_best_fitness = max(fitness_scores)
            print(f"Generation {generation+1} best fitness score: {max(fitness_scores)}")
            # print(f"Generation {generation + 1} avg score is: {round(mean(fitness_scores), 5)}")

            # Check for convergence
            if curr_gen_best_fitness > best_fitness_so_far:
                best_fitness_so_far = curr_gen_best_fitness
                # Reset stuck count if there's improvement
                gen_stuck_count = 0
            else:
                # Increment stuck count if no improvement
                gen_stuck_count += 1

            if gen_stuck_count >= STUCK_THRESHOLD:
                # If no improvement for 20 generations, stop the process
                print("Convergence reached. stuck for ", STUCK_THRESHOLD, " generations")
                break

            # Selecting the top performing networks (elites)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_population = [population[i] for i in sorted_indices[:int(self.population_size * ELITE_SIZE)]]
            # Remaining population after elites have been selected
            remaining_population = list(set(population) - set(elite_population))

            # Creating offspring population via crossover
            offspring_population = []
            num_offsprings = self.population_size - len(elite_population)
            # TODO: if desperate with low accuracy - try where each parent is chosen once for crossover
            for _ in range(num_offsprings):
                parent1 = np.random.choice(remaining_population)
                parent2 = np.random.choice(elite_population)
                offspring = parent1.crossover(parent2)
                offspring_population.append(offspring)

            # Save some offspring untouched for the next gen population
            num_untouched_offspring = int(num_offsprings * OFFSPRING_UNTOUCHED)
            untouched_offspring = offspring_population[:num_untouched_offspring]

            # Mutate the remaining (touched) offspring population
            for offspring in offspring_population[num_untouched_offspring:]:
                offspring.mutate()

            # Combine elites, untouched offspring and mutated offspring to create the next gen population
            population = elite_population + untouched_offspring + offspring_population[num_untouched_offspring:]

            if gen_stuck_count > 3:
                # Lamarckian method:
                print("Lamarckian evolution")
                new_population = []
                for network in population:
                    new_population.append(self.lamarckian_evolution(network, x_train, y_train))
                population = new_population
        
        # evaluate the fitness of the last gen population, and select the network with the best fitness 
        fitness_scores = [evaluate_fitness(network, x_train, y_train) for network in population]
        best_network = population[np.argmax(fitness_scores)]
        return best_network

    def lamarckian_evolution(self, network, x_train, y_train):
        old_fitness = evaluate_fitness(network, x_train, y_train)
        new_network = copy.deepcopy(network)
        for _ in range(LAMARCKIAN_MUTATIONS):
            new_network.mutate()
        new_fitness = evaluate_fitness(new_network, x_train, y_train)
        if new_fitness > old_fitness:
            return new_network
        else:
            return network


# Neural Network Implementation
class Layer:
    def __init__(self, input_size, output_size, activation=0):
        # Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        # self.weights = np.random.randn(input_size, output_size)
        self.activation = activation

    def forward(self, inputs):
        output = np.dot(inputs, self.weights)
        if self.activation:
            output = sigmoid(output)
        return output

    def get_shape(self):
        return self.weights.shape, self.activation


class NeuralNetwork:
    """
        This class represents a simple feed-forward neural network.
        The network consists of several layers, each represented by a Layer object.
        The network uses these layers to transform its input data when the predict() method is called.
        The class also includes crossover() and mutate() methods.
    """
    def __init__(self):
        # List to hold all layers of the neural network
        self.layers = []

    def add_layer(self, layer):
        # Appends a new layer to the network
        self.layers.append(layer)

    def predict(self, inputs):
        # Passes the inputs through each layer of the network
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        # Converts the output of the final layer to binary predictions
        binary_predictions = (outputs > 0.5).astype(int)
        return binary_predictions.flatten()

    def crossover(self, other_network):
        """
            performs a crossover operation between this NeuralNetwork instance and another.
            The crossover combines the weights of the two parents,
            to generate a new offspring network with weights inherited from both parents.
        """
        # Create the new neural network which will be our offspring
        new_network = create_neural_network()

        for i in range(len(self.layers)):
            # For each layer, we decide from which parent to inherit the weights.
            # This is done randomly: we generate a random number and if it's greater than 0.5,
            # we inherit from the current network. otherwise, we inherit from the other network.
            if np.random.rand() > 0.5:
                new_network.layers[i].weights = np.copy(self.layers[i].weights)
            else:
                new_network.layers[i].weights = np.copy(other_network.layers[i].weights)
        return new_network

    def mutate(self):
        """
        The method is used to randomly adjust the weights in the network's layers to introduce variation.
        the mutation process randomly selects a subset of weights in each layer based on the MUTATION_RATE.
        For the selected weights, a random value (pos/neg) is added to introduce variation.
        """
        # for layer in self.layers:
        #     # Generate a mask for the weights to be mutated
        #     mask = np.random.rand(*layer.weights.shape) < MUTATION_RATE
        #     # Add random noise to selected weights
        #     layer.weights[mask] += np.random.randn(*layer.weights.shape)[mask]
        for layer in self.layers:
            # Generate a mask for the weights to be mutated
            mask = np.random.rand(*layer.weights.shape) < MUTATION_RATE
            # Add random values from -1 to 1 to the selected weights
            layer.weights[mask] += np.random.uniform(-1, 1, size=layer.weights.shape)[mask]


# Main
genetic_algorithm = GeneticAlgorithm()
best_network = genetic_algorithm.evolve(x_train, y_train)

# Testing
test_predictions = best_network.predict(x_test)
accuracy = compute_accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {accuracy}")
