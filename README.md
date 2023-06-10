# Genetic Algorithm for Neural Network Optimization
This repository holds the code for Ex3 in Computational Biology Course.

This Python program uses a genetic algorithm to optimize the structure and parameters of a neural network for binary string classification. The goal of the program is to achieve the highest possible classification accuracy on a test dataset.

## About

The program applies a simple genetic algorithm (GA) to a feedforward neural network (NN) with the aim of enhancing the network's classification performance. The genetic algorithm uses a real-valued representation of the NN's weights, and applies standard GA operations such as selection, crossover, and mutation to evolve a population of NNs over multiple generations.

## Structure

The program consists of several key components:

1. **Data preparation functions:** These functions load the data from the provided .txt files, process the data into a suitable format for the NN, and split the data into a training set and a test set.

2. **Neural network classes:** These classes define the structure of the NN and include methods for forward propagation (used during prediction), crossover (used to combine the weights of two parent NNs), and mutation (used to introduce variation into the NN's weights).

3. **Genetic algorithm class:** This class defines the genetic algorithm, including the initialization of a population of NNs, the evolution process over multiple generations, and the selection of the best NN based on its performance on the training data.

4. **Fitness evaluation functions:** These functions assess the fitness of each NN in the population by computing its classification accuracy on the training data.

## Usage

The primary script in this program is `main.py`. When this script is run, it loads the data, applies the genetic algorithm to evolve a population of neural networks, and finally saves the best network from the final generation.

The following global parameters can be adjusted in `main.py` to tune the behavior of the algorithm:

- `POPULATION_SIZE`: The number of neural networks in the population.
- `MUTATION_RATE`: The probability that a given weight in a network will be mutated.
- `GENERATIONS`: The number of generations over which the population will evolve.
- `ELITE_SIZE`: The proportion of the population with the highest fitness scores that will be preserved unchanged in the next generation.
- `INPUT_SIZE`: The number of inputs to the neural network (equal to the length of the binary strings in the data).
- `HIDDEN_SIZE_1` and `HIDDEN_SIZE_2`: The number of nodes in the first and second hidden layers of the network, respectively.
- `OUTPUT_SIZE`: The number of outputs from the neural network (1 for binary classification).

## Results

The program prints out the best fitness score achieved in each generation, as well as the best score achieved in the final generation. These scores represent the highest classification accuracy achieved on the training data by any network in the population.

## Requirements

The program requires:

- Python 3.x 
- NumPy library
- The `copy` module in Python's standard library
- The `statistics` module in Python's standard library
- The `warnings` module in Python's standard library

Please make sure to have these dependencies installed before running the program. You can install NumPy using pip:

```shell
pip install numpy
```

The rest of the required modules are part of Python's standard library and should be available with a standard Python installation.
