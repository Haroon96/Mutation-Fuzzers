from random import choice
from data_generators.data_generator import DataGenerator
from sock_puppets.api import api
from rl_fuzzer.util import is_bug
from rl_fuzzer.embedding import embed, cosine_similarity
import json
import numpy as np

class YouTubeEnv:
    def __init__(self, data_gen, num_videos=30, max_gens=50):
        self.data_gen:DataGenerator = data_gen
        self.max_gens = max_gens
        self.num_videos = num_videos
        with open('rl_fuzzer/data/videos.json') as f:
            self.video_ids = []
            self.video_embs = []
            for vid in json.load(f):
                self.video_ids.append(vid['video_id'])
                self.video_embs.append(vid['embedding'])
            self.video_embs = np.array(self.video_embs)
        self.reset()

    def step(self, action):
        # create next state
        n = len(self.state)
        idx = choice(range(n))
        next_state = self.state[:]
        next_state[idx], next_state[idx] = self.find_closest_video(action)

        # get reward
        recs, reward = self.get_reward(self.state)

        # log actions
        self.actions.append(action)
        self.action_video_ids.append(next_state)
        self.rewards.append(reward)
        self.states.append(next_state)
        self.recommendations.append(recs)

        # update state
        self.state = next_state
        
        # increment steps
        self.steps += 1

        # return next state and reward
        return self.state, reward

    def reset(self):
        sample = self.data_gen.sample_metadata(self.num_videos * 2)
        self.state = []
        self.state_embs = []
        i = 0
        while len(self.state) < self.num_videos:
            video = sample[i]
            if (emb := embed(video['title'])) is not None:
                self.state.append(video['video_id'])
                self.state_embs.append(emb)
            i += 1
            
        self.initial_state = self.state
        self.states = []
        self.actions = []
        self.rewards = []
        self.recommendations = []
        self.action_video_ids = []
        self.steps = 0

    def get_reward(self, trace):
        recs = api(trace, 'youtube.py')
        reward = sum([i for i in recs if is_bug(i)])
        return recs, reward

    def find_closest_video(self, emb):
        similarity = lambda x : cosine_similarity(x, emb)
        most_similar = np.apply_along_axis(similarity, 1, self.video_embs).argmax()
        return self.video_ids[most_similar], self.video_embs[most_similar]

    def is_done(self):
        return self.steps >= self.max_gens