# Genetic Algorithm for Neural Network Optimization
This repository holds the code for Ex3 in Computational Biology Course.

This Python program uses a genetic algorithm to optimize the structure and parameters of a neural network for binary string classification. The goal of the program is to achieve the highest possible classification accuracy on a test dataset.

## About

The program applies a simple genetic algorithm (GA) to a feedforward neural network (NN) with the aim of enhancing the network's classification performance. The genetic algorithm uses a real-valued representation of the NN's weights, and applies standard GA operations such as selection, crossover, and mutation to evolve a population of NNs over multiple generations.

## Features
- Optimizes a Neural Network using Genetic Algorithms.
- Uses rank-based parent selection.
- Utilizes the Lamarckian method to avoid local optima.
- Implements early convergence checks.


## Results

After training the Neural Network using the Genetic Algorithm, the network's weights are saved to a .npz file. You can use these weights to set up the optimized network for further use or testing.

The program also evaluates the performance of the optimized network on a test set and prints out the accuracy score.  
Please note that the actual accuracy might differ depending on the complexity of the task and the quality of the input data. The Genetic Algorithm might not always find the global optimum and can be influenced by factors such as population size, mutation rate, the number of generations, etc.

The best fitness scores for each generation are also tracked and stored in a list (`best_fitness_list`), which you can use to visualize the optimization process. i.e:  


<img width="300" alt="image" src="https://github.com/OrrBav/Genetic-Neural-Network/assets/112930532/cae2dd80-fa2a-4272-835a-4dfd860acb42">


## Prerequisites

The program requires numpy, copy, and random Python libraries. The user needs to supply training and test files for the Neural Network.
