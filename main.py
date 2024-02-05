from argparse import ArgumentParser
import os
from uuid import uuid4
import requests
import json
from strategies import strategies, Strategy
import sock_puppets.api as env
from data_generators.data_generator import DataGenerator
from data_generators.youtube_data_generator import YouTubeVideoGenerator

# default directories
DATA_DIR = os.path.join(os.getcwd(), 'data')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--strategy', choices=list(strategies))
    parser.add_argument('--num-videos', default=30)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--generations', default=1000)
    return parser.parse_args()

def main(args):
    
    # prepare data generator
    data_gen = YouTubeVideoGenerator()

    # pick strategy
    strategy: Strategy = strategies[args.strategy](data_gen)

    # generate run id
    run_id = str(uuid4())[:8]

    # generate initial trace
    trace = data_gen.sample_videos(args.num_videos)

    # run until convergence
    generation = 0
    while generation < args.generations:

        # run trace
        recommendations = env.api(trace, 'youtube.py')

        # post data
        post_data = dict(
            run_id=run_id,
            generation=generation,
            strategy=args.strategy,
            alpha=args.alpha,
            num_videos=args.num_videos,
            training=trace,
            recommendations=recommendations
        )

        # send data to server
        requests.post('http://lake.cs.ucdavis.edu/fuzzerapi/process-data', data=json.dumps(post_data))

        # get new trace
        trace = strategy.mutate(trace, args.alpha)


if __name__ == '__main__':
    args = parse_args()
    main(args)