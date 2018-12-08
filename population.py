from neural_network import NeuralNetwork


class Population(object):
    layers: int
    neurons: int
    population: list

    def __init__(self, layers: int, neurons: int, max_individuals: int):
        self.layers = layers
        self.neurons = neurons
        self.individuals = [NeuralNetwork(361, layers, layers, neurons, neurons)
                            for _ in range(max_individuals)]
        print('layers: {}, neurons: {}'.format(layers, neurons))

        self.bonus = 0
        self.malus = 0

    def calculate_pop_score(self):
        for indiv in self.individuals:
            self.bonus += indiv.bonus
            self.malus += indiv.malus
