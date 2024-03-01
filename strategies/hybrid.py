from data_generators.data_generator import DataGenerator
from strategies.strategy import Strategy
from random import choices, random

class Hybrid(Strategy):
    def __init__(self, data_gen):
        self.data_gen:DataGenerator = data_gen

    def mutate(self, trace, alpha):
        mutated_trace = trace[:]
        n = len(mutated_trace)
        ss = 1 if alpha is None else int(n * alpha)
        if random() <= 0.8:
            sample = self.data_gen.sample_bugs(ss)
        else:
            sample = self.data_gen.sample_videos(ss)
        idx = choices(range(n), k=ss)
        for i, j in enumerate(idx):
            mutated_trace[j] = sample[i]
        return mutated_trace