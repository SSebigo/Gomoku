from numpy import dot, exp, ndarray
from numpy.random import randn
import numpy as np
from random import randint


class NeuralNetwork(object):
    nbr_end_points_layers: int
    min_layers: int
    max_layers: int
    min_neurons: int
    max_neurons: int
    nbr_hidden_layers: int
    nbr_neurons_per_layers: list
    hidden_weights_matrices: list
    output_weights_matrix: list
    hidden_neurons_by_hidden_weights_matrices: list
    hidden_activated_neurons_matrices: list
    output_neurons_by_output_weights_matrix: ndarray
    timeout: bool
    victory: bool
    bonus: int
    malus: int

    def __init__(self, nbr_end_points_layers, min_layers, max_layers, min_neurons, max_neurons):
        # Define HyperParameters
        self.nbr_end_points_layers = nbr_end_points_layers
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        self.nbr_hidden_layers = randint(
            self.min_layers, self.max_layers)

        self.nbr_neurons_per_layers = list()
        for i in range(self.nbr_hidden_layers):
            self.nbr_neurons_per_layers.append(
                randint(self.min_neurons, self.max_neurons))

        self.hidden_weights_matrices = list()

        # weight matrix from input to first hidden layer
        self.input_weights_matrix = randn(
            self.nbr_end_points_layers, self.nbr_neurons_per_layers[0])
        for i in range(self.nbr_hidden_layers-1):
            # weight matrix from n hidden to n+1 hidden layer
            self.hidden_weights_matrices.append(randn(
                self.nbr_neurons_per_layers[i], self.nbr_neurons_per_layers[i+1]))
        # weight matrix from last hidden to output layer
        self.output_weights_matrix = randn(
            self.nbr_neurons_per_layers[self.nbr_hidden_layers-1], self.nbr_end_points_layers)

        self.timeout = False
        self.victory = None
        self.bonus = 0
        self.malus = 0

    def feed_forward(self, inputs):
        # self.input_neurons_by_input_weights_matrix = np.dot(inputs, self.input_weights_matrix)
        # self.input_activated_neurons_matrix = self.relu(self.input_neurons_by_input_weights_matrix)

        self.hidden_neurons_by_hidden_weights_matrices = list()
        self.hidden_activated_neurons_matrices = list()
        for i in range(0, self.nbr_hidden_layers):
            if i == 0:
                self.hidden_neurons_by_hidden_weights_matrices.append(
                    dot(inputs, self.input_weights_matrix))
                self.hidden_activated_neurons_matrices.append(
                    self.relu(self.hidden_neurons_by_hidden_weights_matrices[i]))
            else:
                self.hidden_neurons_by_hidden_weights_matrices.append(dot(
                    self.hidden_activated_neurons_matrices[i-1], self.hidden_weights_matrices[i-1]))
                self.hidden_activated_neurons_matrices.append(
                    self.relu(self.hidden_neurons_by_hidden_weights_matrices[i]))

        self.output_neurons_by_output_weights_matrix = dot(
            self.hidden_activated_neurons_matrices[self.nbr_hidden_layers-1], self.output_weights_matrix)
        return self.softmax(self.output_neurons_by_output_weights_matrix)

    # def back_propagation(self):
    #     pass

    def display_weights(self):
        print('WI: ', self.input_weights_matrix,
              ' ', type(self.input_weights_matrix))
        for wn in self.hidden_weights_matrices:
            print('WH: ', wn, ' ', type(wn))
        print('WO: ', self.output_weights_matrix,
              ' ', type(self.output_weights_matrix))

    def display_nbr_hidden_layers(self):
        print('hidden layers: ', self.nbr_hidden_layers)

    def display_nbr_neurons_per_layers(self):
        print('neurons per layers: ', self.nbr_neurons_per_layers)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def tanh(self, x):
        return 2 * self.sigmoid(2 * x) - 1

    def relu(self, x):
        return x * (x > 0)

    def softmax(self, x):
        x = np.array(x)
        ex = exp(x - np.max(x))
        return ex / ex.sum(axis=0)
 