import time

import Reporter
import numpy as np
import sys
import random as rn


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


class TravellingSalesPersonProblem:
    def __init__(self, numObjects):
        pass
    # TODO


def fitness(population, distanceMatrix):
    fitnessMap = {}
    for path in population:
        fitnessMap[tuple(path)] = computePathFitness(path, distanceMatrix)

    return fitnessMap


def computePathFitness(path, distanceMatrix):
    if len(path) == 1:
        return 0

    fitnessValue = 0

    for i in range(len(path) - 1):
        first = path[i]
        second = path[i + 1]

        temp = distanceMatrix[first][second]

        if fitness == np.inf:
            raise Exception("Infinite value in path")

        fitnessValue += temp

    return fitnessValue


def selection(population, distanceMatrix):
    sublist = rn.choices(population, k=5)
    fitnessMap = fitness(sublist, distanceMatrix)
    best = max(fitnessMap.values())
    return best


# TODO
def recombination(dm, path1, path2):
    print(path1)
    print(path2)
    allSS = findAllSubsequences(path1, path2)
    possibleIndices = set(range(0, len(distanceMatrix)))
    SS_dict = {}
    for SS in allSS:
        for x in SS:
            possibleIndices.remove(x)
        possibleIndices.add(SS[0])
        SS_dict[SS[0]] = SS

    start = rn.choice(tuple(possibleIndices))
    possibleIndices.remove(start)

    pathOffspring = createRandomCycle(start, start, dm, possibleIndices, SS_dict)
    pathOffspring.append(start)
    pathOffspring.reverse()
    for key in SS_dict:
        i = pathOffspring.index(key)
        for x in SS_dict[key]:
            if i > len(pathOffspring)-1:
                if not x == pathOffspring[i]:
                    pathOffspring.insert(i, x)
                i += 1
            else:
                pathOffspring.insert(i, x)
                i += 1
    return pathOffspring

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

    for i in range(0, its):
        offspring = []
        for j in range(0, mu):
            # Selection step
            p1 = selection(ksp, population)
            p2 = selection(ksp, population)

            # Recombination step
            offspring.append(recombination(_, p1, p2))

            # Mutation step
            mutate(ksp, offspring[j])
        for ind in population:
            # Mutation step
            mutate(ksp, ind)

        # Elimination step
        population = elimination(ksp, population, offspring, mu)


def isInfinite(b, j, dm, SS_dict):
    if b in SS_dict:
        if j in SS_dict:
            SS1 = SS_dict[b]
            SS2 = SS_dict[j]
            return dm[SS1[0]][SS2[len(SS2) - 1]] == np.inf
        else:
            SS = SS_dict[b]
            return dm[SS[0]][j] == np.inf
    else:
        if j in SS_dict:
            SS = SS_dict[j]
            return dm[b][SS[len(SS)-1]] == np.inf
        else:
            return dm[b][j] == np.inf


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

    # If not all indices have been visited, choose an extension for the current cycle.
    if len(possibleIndices) > 0:
        toBeAddedLater = set()
        # Keep looking for a new indices as an extension for the current cycle.
        while len(possibleIndices) > 0:
            j = rn.choice(tuple(possibleIndices))
            possibleIndices.remove(j)
            toBeAddedLater.add(j)
            # Keep looking for a possible extension that is connected to the last vertex.
            while isInfinite(b, j, dm, SS_dict):
                if len(possibleIndices) <= 0:
                    return None
                j = rn.choice(tuple(possibleIndices))
                possibleIndices.remove(j)
                toBeAddedLater.add(j)
            possibleIndices = possibleIndices.union(toBeAddedLater)
            possibleIndices.remove(j)
            path = createRandomCycle(a, j, dm, possibleIndices, SS_dict)
            # A path was found, return it!
            if path is not None:
                path.append(j)
                return path
        # No extension possible, return None
        return None
    # If all indices have been visited, check if a cycle was found.
    else:
        if not isInfinite(b, a, dm, SS_dict):
            return []
        else:
            return None


def prev(i, path):
    if i-1 >= 0:
        return i-1
    else:
        return len(path)-1


def nxt(i, path):
    if i + 1 < len(path):
        return i + 1
    else:
        return 0


def findAllSubsequences(path1, path2):
    allSS = []
    for i in range(0, len(path1)-1):
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
        if len(SS) > 1:
            allSS.append(SS)
    return allSS

def isValidHamiltonianCycle(dm, path):
    for i in range(0, len(path)):
        if i+1 > len(path)-1:
            if dm[path[i]][path[0]] == np.inf:
                return False
        else:
            if dm[path[i]][path[i+1]] == np.inf:
                return False
    return True

# Test fitness and selection
file = open('tour50.csv')
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()

population = [
    [0, 1, 2],
    [0, 1, 3]
]
print(fitness(population, distanceMatrix))
print(selection(population, distanceMatrix))

sys.setrecursionlimit(100000)

for i in range(0, 10):
    print(i)
    start1 = rn.randint(0, len(distanceMatrix) - 1)
    possibleIndices1 = set(range(0, len(distanceMatrix)))
    possibleIndices1.remove(start1)

    start2 = rn.randint(0, len(distanceMatrix) - 1)
    possibleIndices2 = set(range(0, len(distanceMatrix)))
    possibleIndices2.remove(start2)

    randomCycle1 = createRandomCycle(start1, start1, distanceMatrix, possibleIndices1, {})
    randomCycle1.append(start1)
    randomCycle2 = createRandomCycle(start2, start2, distanceMatrix, possibleIndices2, {})
    randomCycle2.append(start2)
    newCycle = recombination(distanceMatrix, randomCycle1, randomCycle2)
    if not isValidHamiltonianCycle(distanceMatrix, newCycle):
        print("FALSE")

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