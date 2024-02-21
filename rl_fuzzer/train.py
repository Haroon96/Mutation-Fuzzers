from rl_fuzzer.env import YouTubeEnv
import numpy as np
from data_generators.youtube_data_generator import YouTubeVideoGenerator

data_gen = YouTubeVideoGenerator()

env = YouTubeEnv(data_gen=data_gen, num_videos=2)
action = np.random.rand(300)
env.step(action)