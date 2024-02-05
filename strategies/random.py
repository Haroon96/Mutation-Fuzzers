from data_generators.data_generator import DataGenerator
from strategies.strategy import Strategy
from random import choices

class Random(Strategy):
    def __init__(self, data_gen):
        self.data_gen:DataGenerator = data_gen

    def mutate(self, trace, alpha):
        mutated_trace = trace[:]
        n = len(mutated_trace)
        ss = int(n * alpha)
        sample = self.data_gen.sample_videos(ss)
        idx = choices(range(n), k=ss)
        for i, j in enumerate(idx):
            mutated_trace[j] = sample[i]
        return mutated_trace