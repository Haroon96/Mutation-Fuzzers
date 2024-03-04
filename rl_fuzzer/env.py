from random import choices
from data_generators.data_generator import DataGenerator
from sock_puppets.api import api
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
                self.video_embeddings[vid['video_id']] = vid['embedding']
            self.videos = np.array(self.videos)
        self.reset()

    def step(self, action):
        # create next state
        n = len(self.state)
        k = int(self.alpha * n)
        idx = choices(range(n), k=k)
        next_state = self.state[:]
        closest_videos = self.find_closest_videos(action, k)
        for i, ind in enumerate(idx):
            next_state[ind] = closest_videos[i]

        # update state
        self.state = next_state
        
        # increment steps
        self.steps += 1

        # return next state and reward
        return self.state, self.max_gens <= self.steps

    def reset(self):
        sample = self.data_gen.sample_metadata(self.num_videos * 2)
        self.state = []
        i = 0
        while len(self.state) < self.num_videos:
            video = sample[i]
            if video['title'] is not None and (emb := embed(video['title'])) is not None:
                self.state.append(video['video_id'])
                self.video_embeddings[video['video_id']] = emb
            i += 1
        self.steps = 0

    def embed_state(self, state=None):
        if state is None:
            state = self.state
        return np.array([self.video_embeddings[i] for i in state]).astype(np.float32)

    def get_recommendations(self, state):
        while True:
            try: return api(state, 'youtube.py')
            except: continue

    def find_closest_videos(self, emb, k):
        similarity = lambda x : cosine_similarity(x, emb)
        video_embeddings = np.array([self.video_embeddings[i] for i in self.videos])
        most_similar = np.apply_along_axis(similarity, 1, video_embeddings).argsort()[-k:]
        return self.videos[most_similar]