import unittest

import numpy as np

import r0123456 as ea
from hamilton_cycle import hamiltonCycle


file = open('tour50.csv')
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()

class TestEA(unittest.TestCase):

    def test_fitness(self):
        cycle1 = hamiltonCycle([0, 1, 2])
        cycle2 = hamiltonCycle([0, 1, 3])
        population = [
            cycle1,
            cycle2
        ]
        ea.fitness(population, distanceMatrix)
        self.assertEqual(cycle1.getFitness(), 26361.0)
        self.assertEqual(cycle2.getFitness(), 16138.9)

    def test_selection(self):
        cycle1 = hamiltonCycle([0, 1, 2])
        cycle2 = hamiltonCycle([0, 1, 3])
        population = [
            cycle1,
            cycle2
        ]
        self.assertEqual(ea.selection(population, distanceMatrix), 26361.0)
