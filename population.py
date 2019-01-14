from typing import List
from neural_network import NeuralNetwork


class Population(object):
    _individuals: list
    _layers: int
    _neurons: int
    population: list

    def __init__(self, layers: int, neurons: int, max_individuals: int):
        self._layers = layers
        self._neurons = neurons
        self._individuals = [NeuralNetwork((19*19), 19*19, layers, neurons)
                            for _ in range(max_individuals)]
        print('layers: {}, neurons: {}'.format(layers, neurons))

        self.victory = 0
        self.bonus = 0
        # self.malus = 0

    def calculate_pop_score(self):
        for indiv in self.individuals:
            self.victory += indiv.win
            self.bonus += indiv.bonus
            # self.malus += indiv.malus
        
        # self.victory = -self.victory
    
    @property
    def individuals(self) -> List[NeuralNetwork]:
        return self._individuals

    @property
    def layers(self) -> List[NeuralNetwork]:
        return self._layers

    @property
    def neurons(self) -> List[NeuralNetwork]:
        return self._neurons
