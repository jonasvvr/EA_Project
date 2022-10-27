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
    allSS = findAllSubsequences(path1, path2)
    print(allSS)
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


def mutate(dm, individual, n=2):
    """
    Mutates an individual solution (path) by swapping two (by default) or more
    indices. If resulted path is no longer a Hamilton cycle the process is repeated
    until path satisfies conditions of Hamilton cycle.

    :param individual: path of the individual solution
    :param n: number of nodes to swap
    :return: mutated path of the individual solution
    """

    while True:
        # list containing indexes to swap
        toSwap = []
        for i in range(n):
            randomIndex = rn.randint(0, len(individual) - 1)
            while randomIndex in toSwap:
                randomIndex = rn.randint(0, len(individual) - 1)
            toSwap.append(randomIndex)

        # shuffle list of random indexes
        swapped = toSwap.copy()
        rn.shuffle(swapped)
        # shuffle until the indexes are swapped
        while toSwap == swapped:
            rn.shuffle(swapped)

        # dictionary that remembers all values that will be swapped
        value = {swapped[i]: individual[swapped[i]] for i in range(n)}
        for i in range(n):
            individual[toSwap[i]] = value[swapped[i]]

        # check if path is a cycle
        if isValidHamiltonianCycle(dm, individual):
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
    fitnessOfAll = fitness(combined, dm)

    # sort individuals by fitness (the smaller the fitness the better the solution)
    sortedFitness = dict(sorted(fitnessOfAll.items(), key=lambda x: x[1]))

    # select mu individuals
    selected = list(sortedFitness.keys())[0:mu]
    newPopulation = list(map(list, selected))

    return newPopulation


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
                path.append(j)
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
    if i-1 >= 0:
        return i-1
    else:
        return len(path)-1


def nxt(i, path):
    if i + 1 < len(path):
        return i + 1
    else:
        return 0


def appendSS(allSS, SS):
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
        appendSS(allSS, SS)
    return allSS

def isValidHamiltonianCycle(dm, path):
    if not len(set(path)) == len(dm):
        return False
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
print(mutate(distanceMatrix, population[0], 2))

sys.setrecursionlimit(100000)

for i in range(0,1000):
    start1 = rn.randint(0, len(distanceMatrix) - 1)
    possibleIndices1 = set(range(0, len(distanceMatrix)))
    possibleIndices1.remove(start1)
    randomCycle1 = createRandomCycle(start1, start1, distanceMatrix, possibleIndices1, {})
    randomCycle1.reverse()

    start2 = rn.randint(0, len(distanceMatrix) - 1)
    possibleIndices2 = set(range(0, len(distanceMatrix)))
    possibleIndices2.remove(start2)
    randomCycle2 = createRandomCycle(start2, start2, distanceMatrix, possibleIndices2, {})
    randomCycle2.reverse()

    newCycle = recombination(distanceMatrix, randomCycle1, randomCycle2)
    print(newCycle)
    if not isValidHamiltonianCycle(distanceMatrix, randomCycle1):
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