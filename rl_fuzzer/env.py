from random import choices
from data_generators.data_generator import DataGenerator
from sock_puppets.api import api
from rl_fuzzer.util import is_bug
from rl_fuzzer.embedding import embed, cosine_similarity
import json
import numpy as np

class YouTubeEnv:
    def __init__(self, data_gen, alpha=0.5, num_videos=30, max_gens=50):
        self.data_gen:DataGenerator = data_gen
        self.max_gens = max_gens
        self.num_videos = num_videos
        self.alpha = alpha
        self.videos = []
        self.video_embeddings = {}
        with open('rl_fuzzer/data/videos.json') as f:
            for vid in json.load(f):
                self.videos.append(vid['video_id'])
                self.video_embeddings[vid['video_id']] = vid['embedding'].astype(np.float32)
            self.videos = np.array(self.videos)
        self.reset()

    def step(self, action):
        # create next state
        n = len(self.state)
        idx = choices(range(n), int(self.alpha * n))
        next_state = self.state[:]
        next_state[idx] = self.find_closest_videos(action, 5)

        # update state
        self.state = next_state
        
        # increment steps
        self.steps += 1

        # return next state and reward
        return self.state, self.max_gens <= self.steps

    def reset(self):
        sample = self.data_gen.sample_metadata(self.num_videos * 2)
        self.state = []
        while len(self.state) < self.num_videos:
            video = sample[i]
            if (emb := embed(video['title'])) is not None:
                self.state.append(video['video_id'])
                self.video_embeddings = 
            i += 1
        self.steps = 0

    def get_reward(self, trace):
        recs = api(trace, 'youtube.py')
        reward = len([i for i in recs if is_bug(i)])
        return recs, reward

    def find_closest_videos(self, emb):
        similarity = lambda x : cosine_similarity(x, emb)
        most_similar = np.apply_along_axis(similarity, 1, self.videos).argsort()[-5:]
        return self.videos[most_similar]