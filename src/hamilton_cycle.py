import numpy as np
import random as rn

class hamiltonCycle:
    """
        Store hamilton cycle and its fitness value
    """
    def __init__(self, path: list, fitness: float = 0):
        self.path: tuple = tuple(path)
        self.fitness: float = fitness

    def getPath(self):
        """
        :return: hamilton cycle
        """
        return list(self.path)

    def getPathTuple(self):
        """
        :return: tuple of hamilton cycle
        """
        return self.path

    def getFitness(self):
        """
        :return: fitness of hamilton cycle
        """
        return self.fitness

    def setFitness(self, fitness):
        """
            Set fitness value of hamilton cycle
        :param fitness: fitness value to set
        """
        self.fitness = fitness


def nxt(i, path):
    """
     Return the next element in a cycle.
    """
    new_i = i + 1
    if new_i > len(path) - 1:
        return 0
    else:
        return new_i

def appendSS(allSS, SS):
    """
    Helper function that checks if a subsequence is duplicate and longer. It is possible to encounter
    subsequences of subsequences.
    """
    if len(SS) > 1:
        if len(allSS) > 0:
            for x in SS:
                for diff_SS in allSS:
                    if x in diff_SS:
                        if len(SS) > len(diff_SS):
                            allSS.append(SS)
                            allSS.remove(diff_SS)
                        return
        allSS.append(SS)

def findAllSubsequences(path1, path2):
    """
    Finds all common subseaquences between two paths.
    """
    allSS = []
    passed = set()
    for i in range(0, len(path1) - 1):
        if not path1[i] in passed:
            j = path2.index(path1[i])
            SS = []
            stillSS = True
            len_ss = 0
            while stillSS and len_ss < len(path1) - 1:
                v1 = nxt(i, path1)
                v2 = nxt(j, path2)
                if path1[i] == path2[j]:
                    SS.append(path1[i])
                    i = v1
                    j = v2
                    len_ss += 1
                else:
                    stillSS = False
            if len(SS) > 1:
                for x in SS:
                    passed.add(x)
            appendSS(allSS, SS)
    return allSS

def isInfinite(dm, v1, v2, SS_dict):
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


def createRandomCycle(dm, a, b, possibleIndices, SS_dict):
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
            while isInfinite(dm, b, j, SS_dict):
                if len(tmpInd) <= 0:
                    return None
                j = rn.choice(tuple(tmpInd))
                tmpInd.remove(j)

            possibleIndices.remove(j)
            path = createRandomCycle(dm, a, j, possibleIndices, SS_dict)
            possibleIndices.add(j)
            # A path was found, return it!
            if path is not None:
                path.insert(0, j)
                return path
        # No extension possible, return None
        return None
    # If all indices have been visited, check if a cycle was found.
    else:
        if not isInfinite(dm, b, a, SS_dict):
            return [a]
        else:
            return None

def isValidHamiltonianCycle(dm, path):
    """
    Checks if a cycle is a valid hamiltonian cycle by checking the amount of unique elements and
    it checks if there is a connection between all of them.
    """
    if not len(path) == len(dm):
        return False
    for i in range(0, len(path)):
        if i + 1 > len(path) - 1:
            if dm[path[i]][path[0]] == np.inf:
                return False
        else:
            if dm[path[i]][path[i + 1]] == np.inf:
                return False
    return True