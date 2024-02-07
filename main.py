from argparse import ArgumentParser
import os
from uuid import uuid4
import requests
import json
from strategies import strategies, Strategy
import sock_puppets.api as env
from data_generators.youtube_data_generator import YouTubeVideoGenerator
from socket import gethostname

# default directories
DATA_DIR = os.path.join(os.getcwd(), 'data')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--strategy', required=True, choices=list(strategies))
    parser.add_argument('--num-videos', default=30, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--generations', default=100, type=int)
    parser.add_argument('--initial-trace')
    parser.add_argument('--initial-generation', default=0, type=int)
    parser.add_argument('--run-id')
    parser.add_argument('--is-resume', action='store_true')
    return parser.parse_args()

def main(args):
    
    # prepare data generator
    data_gen = YouTubeVideoGenerator()

    # pick strategy
    strategy: Strategy = strategies[args.strategy](data_gen)

    # check if resuming an older run or starting new
    if args.is_resume:
        run_id = args.run_id
        trace = json.loads(args.initial_trace)
        trace = strategy.mutate(trace, args.alpha)
        generation = args.initial_generation + 1
    else:
        run_id = str(uuid4())[:8]
        trace = data_gen.sample_videos(args.num_videos)
        generation = 0

    # run until hit num of generations
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
            generations=args.generations,
            training=trace,
            recommendations=recommendations
        )

        # send data to server
        requests.post(f'http://lake.cs.ucdavis.edu/fuzzerapi/process-data/{gethostname()}', data=json.dumps(post_data))

        # get new trace
        trace = strategy.mutate(trace, args.alpha)
        
        # increment generation
        generation += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
