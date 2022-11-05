import time
import sys
import random as rn
import numpy as np

import Reporter
from hamilton_cycle import hamiltonCycle


# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        ea = evolutionaryAlgorithm(distance_matrix)

        population = ea.initialization(ea.lam)
        for i in range(0, ea.its):
            offspring = []
            for j in range(0, ea.mu):
                # Selection step
                p1 = ea.selection(population)
                p2 = ea.selection(population)

                # Recombination step
                offspring.append(ea.recombination(p1, p2))

                # Mutation step
                if ea.isMutated():
                    offspring[len(offspring) - 1] = ea.mutate(offspring[len(offspring) - 1])
            for index, ind in enumerate(population):
                # Mutation step
                if ea.isMutated():
                    population[index] = ea.mutate(ind)

                # Elimination step
            population = ea.elimination(population, offspring)
            all_fitness = [x.getFitness() for x in population]

            meanObjective = sum(all_fitness) / len(all_fitness)
            bestObjective = min(all_fitness)
            # bestSolution = population[all_fitness.index(bestObjective)].getPath() TODO
            bestSolution = np.array([1, 2, 3, 4])

            if i % 100 == 0:
                print(i, "Average:", meanObjective)
                print(i, "Best:", bestObjective)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            time_left = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if time_left < 0:
                break

        # Your code here.
        return 0


    #
    # def evolutionaryAlgorithm(dm):
    #     population = initialization(dm, lam)
    #     for i in range(0, its):
    #         offspring = []
    #         for j in range(0, mu):
    #             # Selection step
    #             p1 = selection(population)
    #             p2 = selection(population)
    #
    #             # Recombination step
    #             offspring.append(recombination(dm, p1, p2))
    #
    #             # Mutation step
    #             if isMutated(0.1):
    #                 mutate(dm, offspring[len(offspring) - 1], toMutate, mutationTries)
    #         for ind in population:
    #             # Mutation step
    #             if isMutated(0.1):
    #                 mutate(dm, ind, toMutate, mutationTries)
    #
    #         # Elimination step
    #         population = elimination(population, offspring, lam)
    #         allFitness = [x.getFitness() for x in population]
    #         if i % 100 == 0:
    #             print(i, "Average:", sum(allFitness) / len(allFitness))
    #             print(i, "Best:", min(allFitness))


class evolutionaryAlgorithm:

    def __init__(self, dm):
        self.lam = 100
        self.mu = 100
        self.its = 10000
        self.to_mutate = 3
        self.mutation_tries = 20
        self.k = 5
        self.alph = 0.05
        self.dm = dm

    def initialization(self, lam=100):
        """
        Creates a population of random individual solutions (path and
        fitness of the path). Every individual solution is an object
        of class hamiltonCycle (from hamilton_cycle.py).

        :param lam: number of individuals to create in population
        :return: list of random lam individual solutions
        :raises: Exception when a bad hamiltonian path has been created
        """

        population = []
        for i in range(0, lam):
            # get a hamilton cycle as a path
            start = rn.randint(0, len(self.dm) - 1)
            possibleIndices = set(range(0, len(self.dm)))
            possibleIndices.remove(start)
            individualPath = self.createRandomCycle(start, start, possibleIndices, {})
            if not self.isValidHamiltonianCycle(individualPath):
                raise Exception("The returned path was not an HamiltonianCycle")
            individual = hamiltonCycle(individualPath)
            self.compute_path_fitness(individual)
            population.append(individual)
        return population

    def fitness(self, population: [hamiltonCycle]):
        """
        Computes fitness of each cycle in the population
        :param population: popultation of cycles
        """
        for cycle in population:
            if not isinstance(cycle, hamiltonCycle):
                raise TypeError("The population must be a list of hamilton cycles.")
            self.compute_path_fitness(cycle)

    def compute_path_fitness(self, cycle: hamiltonCycle):
        """
        Computes fitness of a path
        :param cycle: given cycle
        :return: fitness value of path
        """
        path = cycle.getPath()
        if len(path) == 1:
            return

        fitness_value = 0

        for i in range(len(path) - 1):
            first = path[i]
            second = path[i + 1]

            weight = self.dm[first][second]

            if weight == np.inf:
                raise Exception("Infinite value in path")

            fitness_value += weight

        cycle.setFitness(fitness_value)

    def selection(self, population):
        """
        Select cycle from population with k-tournament selection
        :param population: population of cycles
        :return: The selected cycle
        """
        sublist = rn.choices(population, k=self.k)
        all_fitness = [cycle.getFitness() for cycle in sublist]
        best = sublist[all_fitness.index(min(all_fitness))]
        return best

    def recombination(self, ind1, ind2):
        """
        Finds all the common subsequences of two paths and creates a new Hamiltonian Cycle containing said subsequences.

        It starts by removing all elements of the common subsequences (.1)
        and it adds the first element of each subsequences(.2). These newly
        added elements are used as a reference to the full subsequence which is used as key for the subsequence-dictionary.

        The cycle algorithm is able to used the dictionary to create full cycles with the subsequences.(.3) After the cycles are found,
        the element representing the subsequence gets replaced by the subsequence.(.4)

        :param path1: Parent one for recombination
        :param path2: Parent two for recombination
        :return: A new Hamiltionian cycle containing all the common subsequences
        """
        path1 = ind1.getPath()
        path2 = ind2.getPath()
        while True:
            allSS = self.findAllSubsequences(path1, path2)
            possibleIndices = set(range(0, len(self.dm)))
            SS_dict = {}

            # (.1)
            for SS in allSS:
                for x in SS:
                    try:
                        possibleIndices.remove(x)
                    except KeyError:
                        print(SS, allSS)
                        raise Exception("kut")
                # (.2)
                possibleIndices.add(SS[0])
                SS_dict[SS[0]] = SS

            start = rn.choice(tuple(possibleIndices))  # (.3)
            possibleIndices.remove(start)
            pathOffspring = self.createRandomCycle(start, start, possibleIndices, SS_dict)

            # (.4)
            for key in SS_dict:
                i = pathOffspring.index(key)
                for x in SS_dict[key]:
                    if not key == x:
                        pathOffspring.insert(i + 1, x)
                    i += 1
            if self.isValidHamiltonianCycle(pathOffspring):
                individual = hamiltonCycle(pathOffspring)
                self.compute_path_fitness(individual)
                return individual

    def mutate(self, individual):
        """
        Mutates an individual solution (path) by swapping two (by default) or more
        indices. If resulted path is no longer a Hamiltonian cycle the process is repeated
        until path satisfies conditions of Hamiltonian cycle. It's not possible to mutate
        some paths in a way they can still be a Hamiltonian therefore, number of mutation
        tries is specified as a parameter.

        :param individual: individual solution (object of class hamiltonCycle)
        :return: individual solution with mutated path and recalculated fitness
        """

        path = individual.getPath()
        for t in range(0, self.mutation_tries):
            # list containing indexes to swap
            toSwap = []
            for i in range(self.to_mutate):
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
            value = {swapped[i]: path[swapped[i]] for i in range(self.to_mutate)}
            for i in range(self.to_mutate):
                path[toSwap[i]] = value[swapped[i]]

            # check if path is a cycle
            if self.isValidHamiltonianCycle(path):
                individual.path = tuple(path)
                self.compute_path_fitness(individual)
                return individual
        return individual

    def isMutated(self):
        """
        Decides if individual will be mutated. Chance of mutation is defined with parameter alph
        which is set to 0.05 (5%) by default.

        :param alph: probability of mutation (float number between 0 and 1)
        :return: True if individual is to be mutated, False otherwise
        """
        p = rn.random()
        return p <= self.alph

    def elimination(self, population, offspring):
        """
        (λ + μ)-elimination based on fitness - lam best solutions are chosen from
        combined list of individual solutions (population + offspring).

        :param population: list of population individuals
        :param offspring: list of offspring individuals
        :return: new population of individual solutions (length of the returned
        population: lam)
        """

        # calculate fitness of population and offspring
        combined = population + offspring
        fitnessOfAll = ((individual.getPath(), individual.getFitness()) for individual in combined)

        # delete old population
        del combined

        # sort individuals by fitness (the smaller the fitness the better the solution)
        sortedFitness = sorted(fitnessOfAll, key=lambda x: x[1])

        # select mu individuals
        selected = sortedFitness[0:self.lam]
        newPopulation = [hamiltonCycle(individual[0], individual[1]) for individual in selected]

        return newPopulation

    def isInfinite(self, v1, v2, SS_dict):
        """
        Checks if there is a connection between vertices 'v1' and 'v2'. These can both be subsequences.

        :param v1: Vertex one
        :param v2: Vertex two
        :param SS_dict: The dictionary of all subsequences.
        :return: Return true if the vertices are connected otherwise false.
        """
        if v1 in SS_dict:
            if v2 in SS_dict:
                SS1 = SS_dict[v1]
                end_v1 = SS1[len(SS1) - 1]
                SS2 = SS_dict[v2]
                begin_v2 = SS2[0]
                return self.dm[end_v1][begin_v2] == np.inf
            else:
                SS = SS_dict[v1]
                return self.dm[SS[len(SS) - 1]][v2] == np.inf
        else:
            if v2 in SS_dict:
                SS = SS_dict[v2]
                return self.dm[v1][SS[0]] == np.inf
            else:
                return self.dm[v1][v2] == np.inf

    def createRandomCycle(self, a, b, possibleIndices, SS_dict):
        """
        Completes a random cycle, starting from element b

        :param a: The first element of the cycle (or path)
        :param b: The last element of the cycle (or path)
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
                while self.isInfinite(b, j, SS_dict):
                    if len(tmpInd) <= 0:
                        return None
                    j = rn.choice(tuple(tmpInd))
                    tmpInd.remove(j)

                possibleIndices.remove(j)
                path = self.createRandomCycle(a, j, possibleIndices, SS_dict)
                possibleIndices.add(j)
                # A path was found, return it!
                if path is not None:
                    path.insert(0, j)
                    return path
            # No extension possible, return None
            return None
        # If all indices have been visited, check if a cycle was found.
        else:
            if not self.isInfinite(b, a, SS_dict):
                return [a]
            else:
                return None

    def nxt(self, i, path):
        """
         Return the next element in a cycle.
        """
        new_i = i + 1
        if new_i > len(path) - 1:
            return 0
        else:
            return new_i

    def appendSS(self, allSS, SS):
        """
        Helper function that checks if a subsequence is duplicate and longer. It is possible to encounter
        subsequences of subsequences.
        """
        if len(SS) > 1 and len(allSS) > 0:
            for x in SS:
                for diff_SS in allSS:
                    if x in diff_SS:
                        if len(SS) > len(diff_SS):
                            allSS.append(SS)
                        return

    def findAllSubsequences(self, path1, path2):
        """
        Finds all common subseaquences between two paths.
        """
        allSS = []
        for i in range(0, len(path1) - 1):
            j = path2.index(path1[i])
            SS = []
            stillSS = True
            len_ss = 0
            while stillSS and len_ss < len(path1) - 1:
                v1 = self.nxt(i, path1)
                v2 = self.nxt(j, path2)
                if path1[i] == path2[j]:
                    SS.append(path1[i])
                    i = v1
                    j = v2
                    len_ss += 1
                else:
                    stillSS = False
            self.appendSS(allSS, SS)
        return allSS

    def isValidHamiltonianCycle(self, path):
        """
        Checks if a cycle is a valid hamiltonian cycle by checking the amount of unique elements and
        it checks if there is a connection between all of them.
        """
        if not len(path) == len(self.dm):
            return False
        for i in range(0, len(path)):
            if i + 1 > len(path) - 1:
                if self.dm[path[i]][path[0]] == np.inf:
                    return False
            else:
                if self.dm[path[i]][path[i + 1]] == np.inf:
                    return False
        return True


sys.setrecursionlimit(100000)
ea = r0123456()
ea.optimize('./data/tour50.csv')
