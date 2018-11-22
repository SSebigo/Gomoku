from neural_network import neural_network


class genetic_algorithm(object):
    def __init__(self, max_population):
        self.population = []
        self.max_population = max_population

        self.elitism = 0.10
        self.mutation_rate = 0.10

    def initial_population(self):
        for i in range(self.max_population):
            self.population.append(neural_network(361, 1, 100, 2, 100))

    def selection(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def dispay_population_info(self):
        for individual in self.population:
            print('individual informations:\nhidden layers: ', individual.display_nbr_hidden_layers(), '\nneurons per layers: ', individual.display_nbr_neurons_per_layers())
