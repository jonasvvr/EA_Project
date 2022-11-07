import unittest

import numpy as np

from r0123456 import evolutionaryAlgorithm
from hamilton_cycle import hamiltonCycle

file = open('tour50.csv')
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()

ea = evolutionaryAlgorithm(distanceMatrix)


class TestEA(unittest.TestCase):

    def test_fitness(self):
        cycle1 = hamiltonCycle([0, 1, 2])
        cycle2 = hamiltonCycle([0, 1, 3])
        population = [
            cycle1,
            cycle2
        ]
        ea.fitness(population)
        self.assertEqual(cycle1.getFitness(), 26361.0)
        self.assertEqual(cycle2.getFitness(), 16138.9)

    def test_selection(self):
        cycle1 = hamiltonCycle([0, 1, 2], 100)
        cycle2 = hamiltonCycle([0, 1, 3], 1000)
        population = [
            cycle1,
            cycle2
        ]
        self.assertEqual(ea.selection(population), cycle1)
