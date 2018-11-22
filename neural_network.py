import numpy as np
import random


class neural_network(object):
    def __init__(self, nbr_end_points_layers, min_layers, max_layers, min_neurons, max_neurons):
        # Define HyperParameters
        self.nbr_end_points_layers = nbr_end_points_layers
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        self.nbr_hidden_layers = random.randint(
            self.min_layers, self.max_layers)

        self.nbr_neurons_per_layers = []
        for i in range(self.nbr_hidden_layers):
            self.nbr_neurons_per_layers.append(
                random.randint(self.min_neurons, self.max_neurons))

        self.hidden_weights_matrices = []

        # weight matrix from input to first hidden layer
        self.input_weights_matrix = np.random.randn(
            self.nbr_end_points_layers, self.nbr_neurons_per_layers[0])
        for i in range(self.nbr_hidden_layers-1):
            # weight matrix from n hidden to n+1 hidden layer
            self.hidden_weights_matrices.append(np.random.randn(
                self.nbr_neurons_per_layers[i], self.nbr_neurons_per_layers[i+1]))
        # weight matrix from last hidden to output layer
        self.output_weights_matrix = np.random.randn(
            self.nbr_neurons_per_layers[self.nbr_hidden_layers-1], self.nbr_end_points_layers)

    def feed_forward(self, inputs):
        # forward propagation through our network
        # self.input_neurons_by_input_weights_matrix = np.dot(inputs, self.input_weights_matrix)
        # self.input_activated_neurons_matrix = self.relu(self.input_neurons_by_input_weights_matrix)

        self.hidden_neurons_by_hidden_weights_matrices = []
        self.hidden_activated_neurons_matrices = []
        for i in range(0, self.nbr_hidden_layers):
            if i == 0:
                self.hidden_neurons_by_hidden_weights_matrices.append(
                    np.dot(inputs, self.input_weights_matrix))
                self.hidden_activated_neurons_matrices.append(
                    self.relu(self.hidden_neurons_by_hidden_weights_matrices[i]))
            else:
                self.hidden_neurons_by_hidden_weights_matrices.append(np.dot(
                    self.hidden_activated_neurons_matrices[i-1], self.hidden_weights_matrices[i-1]))
                self.hidden_activated_neurons_matrices.append(
                    self.relu(self.hidden_neurons_by_hidden_weights_matrices[i]))

        self.output_neurons_by_output_weights_matrix = np.dot(
            self.hidden_activated_neurons_matrices[self.nbr_hidden_layers-1], self.output_weights_matrix)
        return self.relu(self.output_neurons_by_output_weights_matrix)

    # def back_propagation(self):
    #     pass

    def display_weights(self):
        print('W1: ', self.input_weights_matrix,
              ' ', type(self.input_weights_matrix))
        for wn in self.hidden_weights_matrices:
            print('Wn: ', wn, ' ', type(wn))
        print('W2: ', self.output_weights_matrix,
              ' ', type(self.output_weights_matrix))

    def display_nbr_hidden_layers(self):
        print('hidden layers: ', self.nbr_hidden_layers)

    def display_nbr_neurons_per_layers(self):
        print('neurons per layers: ', self.nbr_neurons_per_layers)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return 2 * self.sigmoid(2 * x) - 1

    def relu(self, x):
        return x * (x > 0)
