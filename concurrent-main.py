from argparse import ArgumentParser
import os
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import json
from strategies import strategies, Strategy
import sock_puppets.api as env
from data_generators.youtube_data_generator import YouTubeVideoGenerator
from socket import gethostname
import requests

# default directories
DATA_DIR = os.path.join(os.getcwd(), 'data')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--strategy', required=True, choices=list(strategies))
    parser.add_argument('--strategy-params')
    parser.add_argument('--num-videos', default=30, type=int)
    parser.add_argument('--alpha', default=None, type=float)
    parser.add_argument('--generations', default=100, type=int)
    parser.add_argument('--initial-trace')
    parser.add_argument('--initial-generation', default=0, type=int)
    parser.add_argument('--run-id')
    parser.add_argument('--is-resume', action='store_true')
    return parser.parse_args()

def get_recommendations(trace):
    while True:
        try: return env.api(trace, 'youtube.py')
        except: continue

def main(args):
    
    # prepare data generator
    data_gen = YouTubeVideoGenerator()

    # pick strategy
    strategy_params = [data_gen]
    if args.strategy_params:
        strategy_params.extend(args.strategy_params.split(','))
    strategy: Strategy = strategies[args.strategy](*strategy_params)

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

    # create mutations
    trace_arr = []
    generation_arr = []
    while generation < args.generations:
        # get new trace
        trace = strategy.mutate(trace, args.alpha)
        trace_arr.append(trace)
        generation_arr.append(generation)
        generation += 1

    # run sock puppets
    futures = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for trace in trace_arr:
            futures.append(executor.submit(get_recommendations, trace))

    # get rewards
    recommendations_arr = [future.result() for future in futures]

    # upload data
    for gen, tr, recs in zip(generation_arr, trace_arr, recommendations_arr):
        
        # post data
        post_data = dict(
            run_id=run_id,
            generation=gen,
            strategy=args.strategy,
            alpha=args.alpha,
            num_videos=args.num_videos,
            generations=args.generations,
            strategy_params=args.strategy_params,
            training=tr,
            recommendations=recs
        )

        # send data to server
        requests.post(f'http://lake.cs.ucdavis.edu/fuzzerapi/process-data/{gethostname()}', data=json.dumps(post_data))


if __name__ == '__main__':
    args = parse_args()
    main(args)
