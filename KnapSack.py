from numpy import random
from numpy import ndarray
import random as rn
import copy

class KnapSackProblem:
    def __init__(self, numObjects):
        self.value = [2*random.random() for _ in range(0, numObjects)]
        self.weight = [2*random.random() for _ in range(0, numObjects)]
        self.capacity = 0.25*sum(self.weight)

class Individual:
    def __init__(self, ksp):
        self.order = random.permutation(len(ksp.value))
        self.alpha = 0.05

def fitness(ksp, ind):
    value = 0
    cap = ksp.capacity
    for i in ind.order:
        if cap - ksp.weight[i] >= 0:
            cap -= ksp.weight[i]
            value += ksp.value[i]
    return value

def selection(ksp, population):
    k = 5
    selected = rn.choices(population, k=k)
    fit = [fitness(ksp, x) for x in selected]
    max_value = max(fit)
    max_index = fit.index(max_value)
    return selected[max_index]

def recombination(ksp, p1, p2):
    NSp1 = inKnapSack(ksp, p1)
    NSp2 = inKnapSack(ksp, p2)
    offspring = NSp1.intersection(NSp2)
    for i in NSp1.symmetric_difference(NSp2):
        if random.random() <= 0.5:
            offspring.add(i)

    rem = set(range(0, len(ksp.value)))
    rem = rem.difference(offspring)

    order = ndarray.tolist(random.permutation(list(offspring))) + ndarray.tolist(random.permutation(list(rem)))
    off = Individual(ksp)
    off.order = order

    beta = 2*random.random() - 0.5
    alpha = p1.alpha + beta*(p2.alpha - p1.alpha)
    off.alpha = alpha
    return off

def mutate(ksp, ind):
    if random.random() < ind.alpha:
        i = random.randint(0, len(ksp.value)-1)
        j = random.randint(0, len(ksp.value) - 1)
        tmp = ind.order[i]
        ind.order[i] = ind.order[j]
        ind.order[j] = tmp

def elimination(kps, population, offspring, mu):
    all = population + offspring
    sorted(all, key=(lambda x: fitness(ksp, x)))
    all.reverse()
    return all[0:mu]

def inKnapSack(ksp, ind):
    cap = ksp.capacity
    knapsack = set()
    for i in ind.order:
        if cap - ksp.weight[i] >= 0:
            cap -= ksp.weight[i]
            knapsack.add(i)
    return knapsack

def evolutionaryAlgorithm(ksp):
    lam = 100
    mu = 100
    its = 100
    population = [Individual(ksp) for _ in range(0, lam)]

    for i in range(0,its):
        offspring = []
        for j in range(0,mu):
            p1 = selection(ksp, population)
            p2 = selection(ksp, population)
            offspring.append(recombination(ksp, p1, p2))
            mutate(ksp, offspring[j])
        for ind in population:
            mutate(ksp, ind)

        population = elimination(ksp, population, offspring, mu)

        fitnesses = [fitness(ksp, x) for x in population]
        print("Mean fitness: " + str(sum(fitnesses)/len(fitnesses)) + "    Best fitness: " + str(max(fitnesses)) )

ksp = KnapSackProblem(1000)
ind = Individual(ksp)


def heurBest(ksp):
    heurOrder = sorted(range(0,len(ksp.value)), key= (lambda x: ksp.value[x]/ksp.weight[x]))
    heurOrder.reverse()
    ind = Individual(ksp)
    ind.order = heurOrder
    return fitness(ksp, ind)


print("Capacity: " + str(ksp.capacity))
print("values : " + str(ksp.value))
print("weights : " + str(ksp.weight))
print("Heuristic best: " + str(heurBest(ksp)))
evolutionaryAlgorithm(ksp)
