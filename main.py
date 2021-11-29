import os
import random
import multiprocessing as mp
import numpy
import time

from job3 import *

P_COUNT = os.cpu_count()

PRINT_EACH_ITERATION = True
PRINT_COST_CALCULATION = False


class AP:
    def __init__(self, path: str):
        with open(path) as f:
            self.costs = list(map(lambda line: list(map(int, line.split())), f.readlines()))
        self.count = len(self.costs)
        for _ in range(self.count):
            self.pheromones = [[1 / self.costs[i][j] for j in range(self.count)] for i in range(self.count)]
        self.colony = [Ant(self) for _ in range(COLONY_POPULATION)]
        self.total_best_ant_cost = None
        self.total_best_ant_assignments = None

    def step(self, k):
        for ant in self.colony:
            ant.assign(k)

    def evaporate(self):
        for i in range(self.count):
            for j in range(self.count):
                self.pheromones[i][j] *= (1 - EVAPORATION_PARAMETER)

    def secret_pheromones(self):
        for ant in self.colony:
            ant.calculate_cost()
            delta = 1 / ant.cost
            for i in range(self.count):
                j = ant.assignments[i]
                self.pheromones[i][j] += delta

            if self.total_best_ant_cost is not None:
                e_delta = ELITISM / self.total_best_ant_cost
                for i in range(self.count):
                    j = self.total_best_ant_assignments[i]
                    self.pheromones[i][j] += e_delta

    def iterate(self):
        if PARALLEL:
            def chunk_assgin(indices, ants, rd):
                for i in range(len(indices)):
                    for k in range(self.count):
                        ants[i].assign(k)
                    rd[indices[i]] = ants[i].assignments

            chunks = numpy.array_split(range(len(self.colony)), P_COUNT)
            manager = mp.Manager()
            rd = manager.dict()
            processes = [
                mp.Process(target=chunk_assgin, args=(chunk, self.colony[chunk[0]:chunk[-1] + 1], rd))
                for chunk in chunks
            ]

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            for i in range(len(self.colony)):
                self.colony[i].assignments = rd[i]
        else:
            for k in range(self.count):
                self.step(k)

        self.evaporate()
        self.secret_pheromones()

    def solve(self):
        i = 0
        stagnancy = 0
        while True:
            self.iterate()
            i += 1

            new_best_ant = min(self.colony, key=lambda ant: ant.cost)
            if self.total_best_ant_cost is not None:
                if self.total_best_ant_cost - new_best_ant.cost < IMPROVEMENT_THRESHOLD:
                    stagnancy += 1
                    if stagnancy >= STAGNANCY_THRESHOLD:
                        break
                else:
                    stagnancy = 0

            if self.total_best_ant_cost is None or new_best_ant.cost < self.total_best_ant_cost:
                self.total_best_ant_cost = new_best_ant.cost
                self.total_best_ant_assignments = new_best_ant.assignments.copy()

            if PRINT_EACH_ITERATION:
                print(f'Iteration {i}: Total best is {self.total_best_ant_cost} while iteration best is {new_best_ant.cost}')
        return [self.total_best_ant_cost, self.total_best_ant_assignments]


class Ant:
    def __init__(self, ap: AP):
        self.ap = ap
        self.assignments = [-1] * self.ap.count
        self.cost = None

    def assign(self, k):
        weights = [
            (self.ap.pheromones[k][i] ** ALPHA) * ((1 / self.ap.costs[k][i]) ** BETA)
            if i not in self.assignments[:k] else 0
            for i in range(self.ap.count)
        ]
        self.assignments[k] = random.choices(range(self.ap.count), weights=weights, k=1)[0]

    def calculate_cost(self):
        self.cost = sum([self.ap.costs[i][self.assignments[i]] for i in range(self.ap.count)])


ap = AP(FILE_PATH)

t_start = time.time()
answer = ap.solve()
t_end = time.time()

print(f'Took: {t_end - t_start:.5f} seconds')
print(f'Cost: {answer[0]}')
print('Assignments:', end=' ')
print(answer[1])
if PRINT_COST_CALCULATION:
    print('+'.join([str(ap.costs[i][answer[1][i]]) for i in range(len(answer[1]))]) + '=' + str(answer[0]))
