import Reporter
import numpy as np
import sys
from numpy import random
from numpy import ndarray
import random as rn
import copy

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		yourConvergenceTestsHere = True
		while( yourConvergenceTestsHere ):
			meanObjective = 0.0
			bestObjective = 0.0
			bestSolution = np.array([1,2,3,4,5])

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

		# Your code here.
		return 0



# class TravellingSalesPersonProblem:
#     def __init__(self, numObjects):
# 		pass
# 		#TODO


def fitness(ksp, ind):
	pass
	# TODO

def selection(ksp, population):
	pass
	# TODO

def recombination(ksp, p1, p2):
	pass
	# TODO

def mutate(ksp, ind):
	pass
	# TODO

def elimination(kps, population, offspring, mu):
	pass
	# TODO

def evolutionaryAlgorithm(ksp):
    lam = 100
    mu = 100
    its = 100
    # population = [TODO for _ in range(0, lam)]

    for i in range(0,its):
        offspring = []
        for j in range(0,mu):
			# Selection step
            p1 = selection(ksp, population)
            p2 = selection(ksp, population)

			#Recombination step
            offspring.append(recombination(ksp, p1, p2))

			# Mutation step
            mutate(ksp, offspring[j])
        for ind in population:
			# Mutation step
            mutate(ksp, ind)

		# Elimination step
        population = elimination(ksp, population, offspring, mu)

def createRandomCycle(a, b, dm, pi):
	if len(pi) > 0:
		possibleIndices = pi.copy()
		toBeAddedLater = []
		while len(possibleIndices) > 0:
			j = rn.choice(possibleIndices)
			possibleIndices.remove(j)
			toBeAddedLater.append(j)
			while dm[b][j] == np.inf:
				if len(possibleIndices) <= 0:
					return None
				j = rn.choice(possibleIndices)
				possibleIndices.remove(j)
				toBeAddedLater.append(j)
			possibleIndices = possibleIndices + toBeAddedLater
			possibleIndices.remove(j)
			path = createRandomCycle(a, j, dm, possibleIndices)
			if path is not None:
				path.append(j)
				return path
	else:
		if dm[a][b] != np.inf:
			return []
		else:
			return None

filename = "tour50.csv"
file = open(filename)
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()

sys.setrecursionlimit(100000)
for i in range(0,1000):
	start = rn.randint(0,len(distanceMatrix)-1)
	possibleIndices = list(range(0,len(distanceMatrix)))
	possibleIndices.remove(start)
	randomCycle = createRandomCycle(start, start, distanceMatrix, possibleIndices)
	randomCycle.append(start)
	print(i)