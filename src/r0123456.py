import time
import sys
import random as rn
import numpy as np
import Reporter
from hamilton_cycle import hamiltonCycle, findAllSubsequences, createRandomCycle, isValidHamiltonianCycle


# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename, lam=100, mu=100, its=10000, to_mutate=3, mutation_tries=20, k=5, alph=0.05):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        ea = evolutionaryAlgorithm(distance_matrix, lam, mu, its, to_mutate, mutation_tries, k, alph)

        population = ea.initialization()
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


class evolutionaryAlgorithm:

    def __init__(self, dm, lam, mu, its, to_mutate, mutation_tries, k, alph):
        self.lam = lam
        self.mu = mu
        self.its = its
        self.to_mutate = to_mutate
        self.mutation_tries = mutation_tries
        self.k = k
        self.alph = alph
        self.dm = dm

    def initialization(self):
        """
        Creates a population of random individual solutions (path and
        fitness of the path). Every individual solution is an object
        of class hamiltonCycle (from hamilton_cycle.py).

        :return: list of random lam individual solutions
        :raises: Exception when a bad hamiltonian path has been created
        """
        population = []
        for _ in range(0, self.lam):
            # get a hamilton cycle as a path
            start = rn.randint(0, len(self.dm) - 1)
            possibleIndices = set(range(0, len(self.dm)))
            possibleIndices.remove(start)
            individualPath = createRandomCycle(self.dm, start, start, possibleIndices, {})
            if not isValidHamiltonianCycle(self.dm, individualPath):
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
        added elements are used as a reference to the full subsequence
        which is used as key for the subsequence-dictionary.

        The cycle algorithm is able to used the dictionary to create full cycles with the subsequences.(.3)
        After the cycles are found, the element representing the subsequence gets replaced by the subsequence.(.4)

        :param ind1: Parent one for recombination
        :param ind2: Parent two for recombination
        :return: A new Hamiltionian cycle containing all the common subsequences
        """
        path1 = ind1.getPath()
        path2 = ind2.getPath()
        while True:
            allSS = findAllSubsequences(path1, path2)
            possibleIndices = set(range(0, len(self.dm)))
            SS_dict = {}

            # (.1)
            for SS in allSS:
                for x in SS:
                    possibleIndices.remove(x)
                # (.2)
                possibleIndices.add(SS[0])
                SS_dict[SS[0]] = SS

            start = rn.choice(tuple(possibleIndices))  # (.3)
            possibleIndices.remove(start)
            pathOffspring = createRandomCycle(self.dm, start, start, possibleIndices, SS_dict)

            # (.4)
            for key in SS_dict:
                i = pathOffspring.index(key)
                for x in SS_dict[key]:
                    if not key == x:
                        pathOffspring.insert(i + 1, x)
                    i += 1

            if isValidHamiltonianCycle(self.dm, pathOffspring):
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
        for _ in range(0, self.mutation_tries):
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
            if isValidHamiltonianCycle(self.dm, path):
                individual.path = tuple(path)
                self.compute_path_fitness(individual)
                return individual
        return individual

    def isMutated(self):
        """
        Decides if individual will be mutated. Chance of mutation is defined with parameter alph
        which is set to 0.05 (5%) by default.

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

    def makeSameSize(self, allSS):
        allSizes = set([len(x) for x in allSS])
        if len(allSS) > 1:
            l = max(allSizes)
            for SS in allSS:
                if len(SS) < l:
                    for i in range(0, l - len(SS)):
                        SS.insert(1, 0)
        return allSS

    def recombinationNumba(self, ind1, ind2):
        """
        Finds all the common subsequences of two paths and creates a new Hamiltonian Cycle containing said subsequences.

        It starts by removing all elements of the common subsequences (.1)
        and it adds the first element of each subsequences(.2). These newly
        added elements are used as a reference to the full subsequence
        which is used as key for the subsequence-dictionary.

        The cycle algorithm is able to used the dictionary to create full cycles with the subsequences.(.3)
        After the cycles are found, the element representing the subsequence gets replaced by the subsequence.(.4)

        :param ind1: Parent one for recombination
        :param ind2: Parent two for recombination
        :return: A new Hamiltionian cycle containing all the common subsequences
        """
        path1 = ind1.getPath()
        path2 = ind2.getPath()
        while True:
            allSS = findAllSubsequences(path1, path2)
            possibleIndices = set(range(0, len(self.dm)))
            SS_dict = {}
            reprSS = []
            reprSS.append(-1)

            # (.1)
            for SS in allSS:
                for x in SS:
                    if not x == -1:
                        possibleIndices.remove(x)
                # (.2)
                possibleIndices.add(SS[0])
                reprSS.append(SS[0])
                SS_dict[SS[0]] = SS

            reprSS = np.array(reprSS)
            allSS = self.makeSameSize(allSS)
            allSS = np.array(allSS)
            start = rn.choice(tuple(possibleIndices))  # (.3)
            possibleIndices.remove(start)
            pathOffspring = testFileNumba.createRandomCycle(self.dm, start, start, possibleIndices, allSS, reprSS)

            # (.4)
            for key in SS_dict:
                i = pathOffspring.index(key)
                for x in SS_dict[key]:
                    if not key == x:
                        pathOffspring.insert(i + 1, x)
                    i += 1

            if isValidHamiltonianCycle(self.dm, pathOffspring):
                individual = hamiltonCycle(pathOffspring)
                self.compute_path_fitness(individual)
                return individual


sys.setrecursionlimit(100000)
ea = r0123456()
ea.optimize('./data/tour50.csv')
