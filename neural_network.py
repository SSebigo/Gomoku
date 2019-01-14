from numpy import dot, exp, ndarray
from numpy.random import randn
from numpy.random import normal
from random import random
from random import randint
from random import uniform
from typing import List
import numpy as np

import sys

from GomokuTournament.sources.Player import Player


class NeuralNetwork(Player):
    _id: int
    _input_layer_size: int
    _output_layer_size: int
    _max_layers: int
    _max_neurons: int
    _nbr_hidden_layers: int
    nbr_neurons_per_layers: list
    _hidden_weights_matrices: list
    hidden_neurons_by_hidden_weights_matrices: list
    hidden_activated_neurons_matrices: list

    def __init__(self, input_layer_size, output_layer_size, max_layers, max_neurons, first_gen=True):
        # Define HyperParameters
        super().__init__()
        self._id = id(self)
        self._input_layer_size = input_layer_size
        self._output_layer_size = output_layer_size
        self._max_layers = max_layers
        self._max_neurons = max_neurons

        self._nbr_hidden_layers = self._max_layers

        self.nbr_neurons_per_layers = list()
        for i in range(self._nbr_hidden_layers):
            self.nbr_neurons_per_layers.append(self._max_neurons)

        self._hidden_weights_matrices = list()
        if first_gen:
            # weight matrix from input to first hidden layer
            self._input_weights_matrix = randn(
                self._input_layer_size, self.nbr_neurons_per_layers[0])
            for i in range(self._nbr_hidden_layers-1):
                # weight matrix from n hidden to n+1 hidden layer
                self._hidden_weights_matrices.append(randn(
                    self.nbr_neurons_per_layers[i], self.nbr_neurons_per_layers[i+1]))
            # weight matrix from last hidden to output layer
            self._output_weights_matrix = randn(
                self.nbr_neurons_per_layers[self._nbr_hidden_layers-1], self._output_layer_size)

    def fill_weights(self, input_weights, hidden_weights, output_weights):
        self._input_weights_matrix = np.array(input_weights)
        for i in range(self._nbr_hidden_layers-1):
            self._hidden_weights_matrices.append(np.array(hidden_weights[i]))
        self._output_weights_matrix = np.array(output_weights)

    def compute(self, board: List[float]) -> List[float]:
        prediction = self.feed_forward(board).tolist()
        return prediction

    def feed_forward(self, inputs):
        self.hidden_neurons_by_hidden_weights_matrices = list()
        self.hidden_activated_neurons_matrices = list()
        for i in range(0, self._nbr_hidden_layers):
            if i == 0:
                self.hidden_neurons_by_hidden_weights_matrices.append(
                    dot(np.array(inputs), self._input_weights_matrix))
                self.hidden_activated_neurons_matrices.append(
                    self.relu(self.hidden_neurons_by_hidden_weights_matrices[i]))
            else:
                self.hidden_neurons_by_hidden_weights_matrices.append(dot(
                    self.hidden_activated_neurons_matrices[i-1], self._hidden_weights_matrices[i-1]))
                self.hidden_activated_neurons_matrices.append(
                    self.relu(self.hidden_neurons_by_hidden_weights_matrices[i]))

        self.output_neurons_by_output_weights_matrix = dot(
            self.hidden_activated_neurons_matrices[self._nbr_hidden_layers-1], self._output_weights_matrix)
        return self.softmax(self.output_neurons_by_output_weights_matrix)

    def mutate(self, mutation_rate):
        layer = randint(0, 2)
        if layer == 0:
            # Mutate Input
            for i in range(self._input_layer_size):
                for j in range(self._max_neurons):
                    if random() <= mutation_rate:
                        replace_perturb = randint(0, 1)
                        if replace_perturb == 0:
                            change = uniform(-1., 1.)
                            self._input_weights_matrix[i][j] = change
                        else:
                            change = normal(0, 1)
                            self._input_weights_matrix[i][j] += change
        elif layer == 1:
            # Mutate Hidden
            h_layer = randint(0, self._nbr_hidden_layers-1)
            # for l in range(self._nbr_hidden_layers-1):
            for i in range(self._max_neurons):
                for j in range(self._max_neurons):
                    if random() <= mutation_rate:
                        replace_perturb = randint(0, 1)
                        if replace_perturb == 0:
                            change = uniform(-1., 1.)
                            self._hidden_weights_matrices[h_layer-1][i][j] = change
                        else:
                            change = normal(0, 1)
                            self._hidden_weights_matrices[h_layer-1][i][j] += change
        else:
            # Mutate Output
            for i in range(self._max_neurons):
                for j in range(self._output_layer_size):
                    if random() <= mutation_rate:
                        replace_perturb = randint(0, 1)
                        if replace_perturb == 0:
                            change = uniform(-1., 1.)
                            self._output_weights_matrix[i][j] = change
                        else:
                            change = normal(0, 1)
                            self._output_weights_matrix[i][j] += change

    def display_weights(self):
        print('WI: ', self._input_weights_matrix,
              ' ', type(self._input_weights_matrix))
        for wn in self._hidden_weights_matrices:
            print('WH: ', wn, ' ', type(wn))
        print('WO: ', self._output_weights_matrix,
              ' ', type(self._output_weights_matrix))

    def display_nbr_hidden_layers(self):
        print('hidden layers: ', self._nbr_hidden_layers)

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

    @property
    def id(self) -> int:
        return self._id

    @property
    def max_layers(self) -> int:
        return self._max_layers

    @property
    def max_neurons(self) -> int:
        return self._max_neurons

    @property
    def input_layer_size(self) -> int:
        return self._input_layer_size

    @property
    def output_layer_size(self) -> int:
        return self._output_layer_size

    @property
    def nbr_hidden_layers(self) -> int:
        return self._nbr_hidden_layers

    @property
    def input_weights_matrix(self) -> int:
        return self._input_weights_matrix

    @property
    def hidden_weights_matrices(self) -> int:
        return self._hidden_weights_matrices

    @property
    def output_weights_matrix(self) -> int:
        return self._output_weights_matrix
