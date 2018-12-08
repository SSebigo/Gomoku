# from genetic_algorithm import genetic_algorithm
from random import choice
from numpy import array, argmax

from genetic_algorithm import GeneticAlgorithm

def main():
    # plateau = list()
    # for i in range(361):
    #     plateau.append(choice([-1,0,1]))
    # # print(array(plateau, dtype=float))
    # nn = NeuralNetwork(361, 100, 100, 100, 100)
    # print(argmax(nn.feed_forward(array(plateau, dtype=float))))
    ga = GeneticAlgorithm(10, 100, 10, 500)

main()