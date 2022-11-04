import time
import sys
import random as rn
import numpy as np

import Reporter
import hamilton_cycle
from hamilton_cycle import hamiltonCycle


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
        while (yourConvergenceTestsHere):
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1, 2, 3, 4, 5])

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


def initialization(dm, lam=100):
    """
    Creates a population of random individual solutions (path and
    fitness of the path). Every individual solution is an object
    of class hamiltonCycle (from hamilton_cycle.py).

    :param dm: distance matrix
    :param lam: number of individuals to create in population
    :return: list of random lam individual solutions
    :raises: Exception when a bad hamiltonian path has been created
    """

    population = []
    for i in range(0, lam):
        # get a hamilton cycle as a path
        start = rn.randint(0, len(dm) - 1)
        possibleIndices = set(range(0, len(dm)))
        possibleIndices.remove(start)
        individualPath = createRandomCycle(start, start, dm, possibleIndices, {})
        if not isValidHamiltonianCycle(dm, individualPath):
            raise Exception("The returned path was not an HamiltonianCycle")
        individual = hamiltonCycle(individualPath)
        compute_path_fitness(individual, dm)
        population.append(individual)
    return population


def fitness(population: [hamiltonCycle], distanceMatrix):
    """
    Computes fitness of each cycle in the population
    :param population: popultation of cycles
    :param distanceMatrix: matrix with distances between nodes
    """
    for cycle in population:
        if not isinstance(cycle, hamiltonCycle):
            raise TypeError("The population must be a list of hamilton cycles.")
        compute_path_fitness(cycle, distanceMatrix)


def compute_path_fitness(cycle: hamiltonCycle, distanceMatrix):
    """
    Computes fitness of a path
    :param cycle: given cycle
    :param distanceMatrix: matrix with distances between nodes
    :return: fitness value of path
    """
    path = cycle.getPath()
    if len(path) == 1:
        return 0

    fitness_value = 0

    for i in range(len(path) - 1):
        first = path[i]
        second = path[i + 1]

        weight = distanceMatrix[first][second]

        if weight == np.inf:
            raise Exception("Infinite value in path")

        fitness_value += weight

    cycle.setFitness(fitness_value)


def selection(population):
    """
    Select cycle from population with k-tournament selection
    :param population: population of cycles
    :return: The selected cycle
    """
    sublist = rn.choices(population, k=5)
    all_fitness = [cycle.getFitness() for cycle in sublist]
    best = sublist[all_fitness.index(min(all_fitness))]
    return best


def recombination(dm, ind1, ind2):
    """
    Finds all the common subsequences of two paths and creates a new Hamiltonian Cycle containing said subsequences.

    It starts by removing all elements of the common subsequences (.1) and it adds the first element of each subsequences(.2). These newly
    added elements are used as a reference to the full subsequence which is used as key for the subsequence-dictionary.

    The cycle algorithm is able to used the dictionary to create full cycles with the subsequences.(.3) After the cycles are found,
    the element representing the subsequence gets replaced by the subsequence.(.4)

    :param dm: matrix with distances between nodes
    :param path1: Parent one for recombination
    :param path2: Parent two for recombination
    :return: A new Hamiltionian cycle containing all the common subsequences
    """
    path1 = ind1.getPath()
    path2 = ind2.getPath()
    while True:
        allSS = findAllSubsequences(path1, path2)
        possibleIndices = set(range(0, len(distanceMatrix)))
        SS_dict = {}

        # (.1)
        for SS in allSS:
            for x in SS:
                possibleIndices.remove(x)
            # (.2)
            possibleIndices.add(SS[0])
            SS_dict[SS[0]] = SS

        start = rn.choice(tuple(possibleIndices))   # (.3)
        possibleIndices.remove(start)
        pathOffspring = createRandomCycle(start, start, dm, possibleIndices, SS_dict)

        # (.4)
        for key in SS_dict:
            i = pathOffspring.index(key)
            for x in SS_dict[key]:
                if i > len(pathOffspring) - 1:
                    if not x == pathOffspring[i]:
                        pathOffspring.insert(i, x)
                    i += 1
                else:
                    pathOffspring.insert(i+1, x)
                    i += 1
        if isValidHamiltonianCycle(dm, pathOffspring):
            individual = hamiltonCycle(pathOffspring)
            compute_path_fitness(individual, dm)
            return individual


def mutate(dm, individual, n=2):
    """
    Mutates an individual solution (path) by swapping two (by default) or more
    indices. If resulted path is no longer a Hamilton cycle the process is repeated
    until path satisfies conditions of Hamilton cycle.

    :param individual: individual solution (object of class hamiltonCycle)
    :param n: number of nodes to swap
    :return: individual solution with mutated path and recalculated fitness
    """

    path = individual.getPath()
    while True:
        # list containing indexes to swap
        toSwap = []
        for i in range(n):
            randomIndex = rn.randint(0, len(path) - 1)
            while randomIndex in toSwap:
                randomIndex = rn.randint(0, len(path) - 1)
            toSwap.append(randomIndex)

        # shuffle list of random indexes
        swapped = toSwap.copy()
        rn.shuffle(swapped)
        # shuffle until the indexes are swapped
        while toSwap == swapped:
            rn.shuffle(swapped)

        # dictionary that remembers all values that will be swapped
        value = {swapped[i]: path[swapped[i]] for i in range(n)}
        for i in range(n):
            path[toSwap[i]] = value[swapped[i]]

        # check if path is a cycle
        if isValidHamiltonianCycle(dm, path):
            individual.path = tuple(path)
            compute_path_fitness(individual, dm)
            return individual


def elimination(dm, population, offspring, mu):
    """
    (λ + μ)-elimination based on fitness - mu best solutions are chosen from
    combined list of individual solutions (population + offspring).

    :param population: list of population individuals
    :param offspring: list of offspring individuals
    :param mu: number of solutions left after elimination
    :return: new population of individual solutions (length of the returned
    population: mu)
    """

    # calculate fitness of population and offspring
    combined = population + offspring
    fitnessOfAll = ((individual.getPath(), individual.getFitness()) for individual in combined)

    # delete old population
    del combined

    # sort individuals by fitness (the smaller the fitness the better the solution)
    sortedFitness = sorted(fitnessOfAll, key=lambda x: x[1])

    # select mu individuals
    selected = sortedFitness[0:mu]
    newPopulation = [hamiltonCycle(individual[0], individual[1]) for individual in selected]

    return newPopulation


def evolutionaryAlgorithm(dm):
    lam = 100
    mu = 10
    its = 1000
    population = initialization(dm, lam)
    for i in range(0, its):
        offspring = []
        for j in range(0, mu):
            # Selection step
            p1 = selection(population)
            p2 = selection(population)

            # Recombination step
            offspring.append(recombination(dm, p1, p2))

            # Mutation step
            # mutate(dm, offspring[len(offspring)-1], n=2)
        # for ind in population:
            # Mutation step
            # mutate(dm, ind, n=2)

        # Elimination step
        population = elimination(dm, population, offspring, mu)
        allFitness = [x.getFitness() for x in population]
        if i % 10 == 0:
            print(i, "Average:", sum(allFitness)/len(allFitness))
            print(i, "Best:", sum(allFitness) / len(allFitness))


def isInfinite(v1, v2, dm, SS_dict):
    """
    Checks if there is a connection between vertices 'v1' and 'v2'. These can both be subsequences.

    :param v1: Vertex one
    :param v2: Vertex two
    :param dm: The distance matrix
    :param SS_dict: The dictionary of all subsequences.
    :return: Return true if the vertices are connected otherwise false.
    """
    if v1 in SS_dict:
        if v2 in SS_dict:
            SS1 = SS_dict[v1]
            end_v1 = SS1[len(SS1) - 1]
            SS2 = SS_dict[v2]
            begin_v2 = SS2[0]
            return dm[end_v1][begin_v2] == np.inf
        else:
            SS = SS_dict[v1]
            return dm[SS[len(SS) - 1]][v2] == np.inf
    else:
        if v2 in SS_dict:
            SS = SS_dict[v2]
            return dm[v1][SS[0]] == np.inf
        else:
            return dm[v1][v2] == np.inf


def createRandomCycle(a, b, dm, possibleIndices, SS_dict):
    """
    Completes a random cycle, starting from element b

    :param a: The first element of the cycle (or path)
    :param b: The last element of the cycle (or path)
    :param dm: The distance matrix
    :param possibleIndices: All possible indices that haven't been visited yet
    :return: A cycle if one was found, otherwise return None and move back up in
    the recursion tree.
    """
    alreadyPassed = set()
    tmpInd = possibleIndices.difference(alreadyPassed)
    # If not all indices have been visited, choose an extension for the current cycle.
    if len(tmpInd) > 0:
        # Keep looking for a new indices as an extension for the current cycle.
        while len(tmpInd) > 0:
            j = rn.choice(tuple(tmpInd))
            tmpInd.remove(j)

            # Keep looking for a possible extension that is connected to the last vertex.
            while isInfinite(b, j, dm, SS_dict):
                if len(tmpInd) <= 0:
                    return None
                j = rn.choice(tuple(tmpInd))
                tmpInd.remove(j)

            possibleIndices.remove(j)
            path = createRandomCycle(a, j, dm, possibleIndices, SS_dict)
            possibleIndices.add(j)
            # A path was found, return it!
            if path is not None:
                path.insert(0,j)
                return path
        # No extension possible, return None
        return None
    # If all indices have been visited, check if a cycle was found.
    else:
        if not isInfinite(b, a, dm, SS_dict):
            return [a]
        else:
            return None


def prev(i, path):
    """
    Return the previous element in a cycle.
    """
    if i - 1 >= 0:
        return i - 1
    else:
        return len(path) - 1


def nxt(i, path):
    """
     Return the next element in a cycle.
    """
    if i + 1 < len(path):
        return i + 1
    else:
        return 0


def appendSS(allSS, SS):
    """
    Helper function that checks if a subsequence is duplicate and longer. It is possible to encounter
    subsequences of subsequences.
    """
    if len(SS) > 1:
        uniqueSS = True
        sameSS = []
        k = 0
        while k < len(SS) and uniqueSS:
            i = 0
            while i < len(allSS) and uniqueSS:
                j = 0
                while j < len(allSS[i]) and uniqueSS:
                    if SS[k] == allSS[i][j]:
                        uniqueSS = False
                        sameSS = allSS[i]
                    j += 1
                i += 1
            k += 1
        if uniqueSS:
            allSS.append(SS)
        else:
            if len(sameSS) < len(SS):
                allSS.append(SS)


def findAllSubsequences(path1, path2):
    """
    Finds all common subseaquences between two paths.
    """
    allSS = []
    for i in range(0, len(path1) - 1):
        j = path2.index(path1[i])
        SS = [path1[i]]
        if i == 0:
            stillSS = True
            tmp_i = i
            tmp_j = j
            while stillSS:
                v1 = prev(tmp_i, path1)
                v2 = prev(tmp_j, path1)
                if path1[v1] == path2[v2]:
                    tmp_i = v1
                    tmp_j = v2
                    SS.append(path1[v1])
                else:
                    stillSS = False
            SS.reverse()
        stillSS = True
        while stillSS:
            v1 = nxt(i, path1)
            v2 = nxt(j, path2)
            if path1[v1] == path2[v2]:
                i = v1
                j = v2
                SS.append(path1[v1])
            else:
                stillSS = False
        appendSS(allSS, SS)
    return allSS


def isValidHamiltonianCycle(dm, path):
    """
    Checks if a cycle is a valid hamiltonian cycle by checking the amount of unique elements and
    it checks if there is a connection between all of them.
    """
    if not len(set(path)) == len(dm):
        return False
    for i in range(0, len(path)):
        if i + 1 > len(path) - 1:
            if dm[path[i]][path[0]] == np.inf:
                return False
        else:
            if dm[path[i]][path[i + 1]] == np.inf:
                return False
    return True


file = open('tour50.csv')
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()

evolutionaryAlgorithm(distanceMatrix)


# # testing initialization
# print("\nInitialization:")
# p = initialization(distanceMatrix, 5)
# for ind in p:
#     print(ind.fitness, ind.path)
#
# # testing mutation
# print("\nMutation:")
# print(p[0].getFitness(), p[0].getPath())
# p[0] = mutate(distanceMatrix, p[0], 3)
# print(p[0].getFitness(), p[0].getPath())
#
# # testing elimination
# while True:
#     newPath = recombination(distanceMatrix, p[0].getPath(), p[1].getPath())
#     if isValidHamiltonianCycle(distanceMatrix, newPath):
#         break
# newInd = hamiltonCycle(newPath, 0)
# compute_path_fitness(newInd, distanceMatrix)
#
# print("\nNew individual:")
# print(newInd.getFitness(), newInd.getPath())
# afterElimination = elimination(distanceMatrix, p, [newInd], 5)
#
# print("\nElimination:")
# for ind in afterElimination:
#     print(ind.fitness, ind.path)

# sys.setrecursionlimit(100000)

# for i in range(0,10000):
#     start1 = rn.randint(0, len(distanceMatrix) - 1)
#     possibleIndices1 = set(range(0, len(distanceMatrix)))
#     possibleIndices1.remove(start1)
#     randomCycle1 = createRandomCycle(start1, start1, distanceMatrix, possibleIndices1, {})
#
#     start2 = rn.randint(0, len(distanceMatrix) - 1)
#     possibleIndices2 = set(range(0, len(distanceMatrix)))
#     possibleIndices2.remove(start2)
#     randomCycle2 = createRandomCycle(start2, start2, distanceMatrix, possibleIndices2, {})
#     newCycle = recombination(distanceMatrix, randomCycle1, randomCycle2)


# path1 = [28, 25, 44, 47, 41, 19, 24, 6, 30, 31, 0, 18, 16, 17, 48, 20, 2, 37, 7, 13, 11, 39, 3, 40, 35, 22, 9, 27, 32, 8, 4, 42, 5, 34, 36, 10, 14, 1, 15, 38, 43, 49, 21, 45, 33, 26, 46, 29, 12, 23]
# path2 = [49, 14, 16, 18, 34, 41, 29, 7, 32, 42, 37, 5, 12, 24, 39, 33, 26, 2, 21, 31, 43, 27, 45, 47, 9, 46, 23, 38, 44, 30, 8, 22, 40, 17, 1, 11, 6, 4, 19, 10, 13, 15, 36, 48, 0, 20, 28, 35, 25, 3]
# print(isValidHamiltonianCycle(distanceMatrix, path2))
# print(isValidHamiltonianCycle(distanceMatrix, path1))
# result = findAllSubsequences(path1, path2)
# newpath = recombination(distanceMatrix, path1, path2)
# print(result)
# print(newpath)
# print(len(newpath))
# print(isValidHamiltonianCycle(distanceMatrix, newpath))
