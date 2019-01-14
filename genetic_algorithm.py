from json import dump
from json import load
from math import floor
from operator import attrgetter
from random import choice
from random import randint
from random import random
from random import sample
from random import uniform
from typing import List
import sys

from neural_network import NeuralNetwork
from population import Population
from GomokuTournament.sources.Tournament import Tournament


def file_exists(filename):
    try:
        with open(filename, 'r') as f:
            return True
    except IOError as x:
        return False


def write_data(file_name, wtype, data, first_write=True):
    with open(file_name, wtype) as out_file:
        dump(data, out_file, indent=4)


def all_the_same(training):
    return training[1:] == training[:-1]


class GeneticAlgorithm(object):
    elitism: float
    max_generations: int
    max_individuals: int
    max_populations: int
    mutation_rate: float
    population: list
    populations: list

    def __init__(self, mutation_rate: int, max_generations: int, max_populations: int, max_individuals: int, max_layers: int, max_neurons: int, from_memory=False):
        self.max_generations = max_generations
        self.max_populations = max_populations
        self.max_individuals = max_individuals
        self._input_layer_size = (19*19)
        self._output_layer_size = 19*19
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.elitism = .20
        self.mutation_rate = mutation_rate
        self.mutation_index = 0
        self.nbr_select_indiv = floor(self.elitism * self.max_individuals)
        self.from_memory = from_memory

        self.populations = list()

        self.current_max_bonus = 0

    def run(self):
        print('====== Running Population tournament! ======')
        # self.run_prime_tournament()
        print('====== NotImplemented ======')
        print('====== Population tournament finished! ======\n')
        self.population = list()
        self.max_layers = 3  # randint(1, self.max_layers)
        self.max_neurons = 250  # randint(1, self.max_neurons)

        if self.from_memory is False:
            self.population = [NeuralNetwork(self._input_layer_size, self._output_layer_size,
                                             self.max_layers, self.max_neurons) for _ in range(self.max_individuals)]
        else:
            for i in range(self.max_individuals):
                if file_exists('data6/weights_'+str(i)+'.json'):
                    with open('data6/weights_'+str(i)+'.json') as file:
                        data = load(file)
                        new_indiv = NeuralNetwork(self._input_layer_size, self._output_layer_size,
                                                  self.max_layers, self.max_neurons, first_gen=False)
                        new_indiv.fill_weights(
                            data['input_weights_matrix'], data['hidden_weights_matrices'], data['output_weights_matrix'])
                        self.population.append(new_indiv)

        start = False
        while start is False:
            tournament = Tournament(self.population)
            tournament.start()
            if sorted(self.population, key=attrgetter('bonus'), reverse=True)[0].bonus > 0.01:
                start = True
            else:
                self.population = [NeuralNetwork(self._input_layer_size, self._output_layer_size,
                                                 self.max_layers, self.max_neurons) for _ in range(self.max_individuals)]

        print('====== Individuals training start! ======\n')
        for i in range(self.max_generations):
            print(f'====== Generation {i} ======')
            if self.darwin_natural_selection() == 1:
                break
        print('====== Individuals training finished! ======\n')

    def darwin_natural_selection(self):
        # Run tournament
        tournament = Tournament(self.population)
        tournament.start()

        max_win = sorted(self.population, key=attrgetter(
            'bonus'), reverse=True)[0].win
        # Retrieve max fitness score
        max_bonus = sorted(self.population, key=attrgetter(
            'bonus'), reverse=True)[0].bonus
        print(f'max bonus: {max_bonus}')

        if self.mutation_index >= len(self.mutation_rate):
            if max_bonus >= 98*(self.mutation_index+1):
                self.mutation_index += 1

        # Backup fittest individuals
        print('====== Backing fittest individuals! ======')
        if max_bonus > self.current_max_bonus:
            self.current_max_bonus = max_bonus
            with open('test_mutation.txt', 'a') as out_file:
                out_file.write('max bonus: '+str(max_bonus)+'\n')
            for i in range(self.max_individuals):
                data = {}
                data['input_layer_size'] = self.population[i].input_layer_size
                data['output_layer_size'] = self.population[i].output_layer_size
                data['max_layers'] = self.population[i].max_layers
                data['max_neurons'] = self.population[i].max_neurons
                data['win'] = self.population[i].win
                data['bonus'] = self.population[i].bonus
                data['input_weights_matrix'] = self.population[i].input_weights_matrix.tolist()
                data['hidden_weights_matrices'] = list()
                for j in range(0, len(self.population[i].hidden_weights_matrices)):
                    data['hidden_weights_matrices'].append(
                        self.population[i].hidden_weights_matrices[j].tolist())
                data['output_weights_matrix'] = self.population[i].output_weights_matrix.tolist()
                write_data('data6/weights_'+str(i)+'.json', 'w+', data)
        print('====== Backup finished! ======')

        tmp_fittest_individuals = sorted(self.population, key=attrgetter('bonus'), reverse=True)

        for i in range(self.nbr_select_indiv):
            print(
                f'win: {tmp_fittest_individuals[i].win} | bonus: {tmp_fittest_individuals[i].bonus} | id: {id(tmp_fittest_individuals[i])}')

        fittest_individuals = list()
        for i in range(0, self.nbr_select_indiv):
            new_indiv = NeuralNetwork(self._input_layer_size, self._output_layer_size, self.max_layers, self.max_neurons, first_gen=False)
            new_indiv.fill_weights(tmp_fittest_individuals[i].input_weights_matrix, tmp_fittest_individuals[
                                   i].hidden_weights_matrices, tmp_fittest_individuals[i].output_weights_matrix)
            new_indiv.bonus = tmp_fittest_individuals[i].bonus
            fittest_individuals.append(new_indiv)

        # Create new population
        print(f'mutation rate: {self.mutation_rate[self.mutation_index]}')
        new_population = list()
        for i in range(self.max_individuals):
            partner1 = self.accept_reject(max_bonus, fittest_individuals)
            # print(f'partner 1: {partner1.id}')
            partner2 = self.accept_reject(max_bonus, fittest_individuals)
            # print(f'partner 2: {partner2.id}')
            child = self.crossover(partner1, partner2)
            child.mutate(self.mutation_rate[self.mutation_index])
            new_population.append(child)
        self.population = new_population
        return max_win

    def accept_reject(self, max_bonus, fittest_individuals) -> NeuralNetwork:
        safe = 0
        while True:
            index = randint(0, self.nbr_select_indiv-1)
            partner = fittest_individuals[index]
            r = uniform(0, max_bonus)
            if r < partner.bonus:
                return partner
            safe += 1
            if safe > 10_000:
                return None

    def crossover(self, partner1, partner2) -> NeuralNetwork:
        crossover_weights = random()
        new_input_matrix = (crossover_weights * partner1.input_weights_matrix) + \
            ((1 - crossover_weights) * partner2.input_weights_matrix)
        new_hidden_matrices = list()
        for i in range(0, self.max_layers-1):
            new_hidden_matrices.append((crossover_weights * partner1.hidden_weights_matrices[i]) + (
                (1 - crossover_weights) * partner2.hidden_weights_matrices[i]))
        new_output_matrix = (crossover_weights * partner1.output_weights_matrix) + (
            (1 - crossover_weights) * partner2.output_weights_matrix)

        new_indiv = NeuralNetwork(self._input_layer_size, self._output_layer_size,
                                  self.max_layers, self.max_neurons, first_gen=False)
        new_indiv.fill_weights(
            new_input_matrix, new_hidden_matrices, new_output_matrix)
        return new_indiv

