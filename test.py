from charm4py import charm, Chare, Group, Reducer
import numpy as np

DATA_SIZE = 100
NUM_ITER = 20

class A(Chare):

    def __init__(self):
        self.data = np.zeros(DATA_SIZE)
        self.iteration = 0

    def work(self):
        # ... do some computation, modifying self.data ...
        # do reduction and send result to element 0
        self.reduce(self.thisProxy[0].collectResult, self.data, Reducer.sum)

    def collectResult(self, result):
        # ... do something with result ...
        self.iteration += 1
        print(self.iteration)
        if self.iteration == NUM_ITER:
            exit()
        else:
            # continue doing work
            self.thisProxy.work()

def main(args):
    g = Group(A)
    g.work()

charm.start(main)