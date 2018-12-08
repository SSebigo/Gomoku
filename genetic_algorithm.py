from math import floor
from operator import attrgetter
from random import choice
from random import randint

from population import Population


class GeneticAlgorithm(object):
    populations: list
    population: list
    fittest_individuals: list
    max_population: int
    elitism: float
    mutation_rate: float

    def __init__(self, max_populations: int, max_individuals: int, max_layers: int, max_neurons: int):
        self.max_populations = max_populations
        self.max_individuals = max_individuals
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.pop_elitism
        self.elitism = 0.10
        self.mutation_rate = 0.10
        self.nbr_select_indiv = floor(self.elitism * self.max_individuals)

        # tuple(number of layers, number of neurons per layers, a list of the individuals)
        self.populations = list()

    def run_pop_tournament(self):
        for _ in range(self.max_populations):
            new_pop = Population(randint(1, self.max_layers), randint(
                1, self.max_neurons), self.max_individuals)
            new_pop.calculate_pop_score()
            self.populations.append(new_pop)

    def select_fittest_attributes(self):
        pass

    def select_fittest_individuals(self):
        # select fittest individuals
        # for indiv in self.population:
        #     indiv.timeout = choice([False, True])
        # filter_timeout_indiv = [
        #     indiv for indiv in self.population if (indiv.timeout is not True)]
        # print(len(filter_timeout_indiv))
        # for indiv in filter_timeout_indiv:
        #     print('individual informations:\nhidden layers: {}\nneurons per layers: {}'
        #           .format(indiv.display_nbr_hidden_layers(),
        #                   indiv.display_nbr_neurons_per_layers()))
        tmp_fittest_individuals = sorted(self.population,
                                         key=attrgetter('fitness'),
                                         reverse=True)
        self.fittest_individuals = tmp_fittest_individuals[:self.nbr_select_indiv]
        for indiv in self.fittest_individuals:
            indiv.fitness = 0

    def crossover(self):
        nbr_new_individual = self.max_population - self.nbr_select_indiv
        # for i in range():

    def mutate(self, indiv):
        # layer to mutate
        mutate_layer = randint(0, indiv.max_layers-1)
        # numbers of values to mutate
        if mutate_layer == 0:
            pass
        elif mutate_layer == indiv.max_layers-1:
            pass
        else:
            pass
        nbr_mutation = floor(self.mutation_rate * )
        # x and y pos of values

    def dispay_population_info(self):
        for individual in self.population:
            print('individual informations:\nhidden layers: {}\nneurons per layers: {}'
                  .format(individual.display_nbr_hidden_layers(),
                          individual.display_nbr_neurons_per_layers()))
        # print(self.population[0].display_weights())
