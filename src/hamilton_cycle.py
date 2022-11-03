
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
