from data_generators.data_generator import DataGenerator
from rl_fuzzer.rl_model import Actor as RLModel
from rl_fuzzer.embedding import embed, cosine_similarity
from strategies.strategy import Strategy
from random import choices
import numpy as np
import json
import torch

class RL(Strategy):
    def __init__(self, data_gen, state_dict='rl_fuzzer/saved_weights/actor_0.weights'):
        self.data_gen:DataGenerator = data_gen
        self.model = RLModel(weights=state_dict)
        self.videos = []
        self.video_embeddings = {}
        with open('rl_fuzzer/data/videos.json') as f:
            for vid in json.load(f):
                self.videos.append(vid['video_id'])
                self.video_embeddings[vid['video_id']] = vid['embedding']
            self.videos = np.array(self.videos)

    def get_title(self, video_id):
        return self.data_gen.get_metadata(video_id)['title']

    def mutate(self, trace, alpha):
        titles = [self.get_title(i) for i in trace]
        state = np.mean([embed(i) for i in titles], axis=0).astype(np.float32)
        state = torch.from_numpy(state)
        n = len(trace)
        k = int(alpha * n)
        idx = choices(range(n), k=k)
        mutated_trace = trace[:]
        with torch.no_grad():
            action = self.model(state)
        closest_videos = self.find_closest_videos(action, k)
        for i, ind in enumerate(idx):
            mutated_trace[ind] = closest_videos[i]
        return mutated_trace
    
    def find_closest_videos(self, emb, k):
        similarity = lambda x : cosine_similarity(x, emb)
        video_embeddings = np.array([self.video_embeddings[i] for i in self.videos])
        most_similar = np.apply_along_axis(similarity, 1, video_embeddings).argsort()[-k:]
        return self.videos[most_similar]