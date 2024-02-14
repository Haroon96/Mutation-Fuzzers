import subprocess
from argparse import ArgumentParser
import sys
from time import sleep
from random import randint
import requests
from socket import gethostname

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-mp', '--max-parallel', type=int, default=5)
    parser.add_argument('-wt', '--wait-time', type=float, default=5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    processes = []
    
    r = requests.get(f'http://lake.cs.ucdavis.edu/fuzzerapi/unfinished-runs/{gethostname()}')
    unfinished_runs = r.json()
    print("# of unfinished runs", len(unfinished_runs))
    
    for run in unfinished_runs:
        # sleep if max processes
        while len(processes) >= args.max_parallel:
            sleep(args.wait_time * 60)
            
            # find and remove finished processes
            for process in processes:
                if process.poll() is not None:
                    processes.remove(process)
        
        # spawn process
        run_id = run['run_id']
        strategy = run['strategy']
        alpha = str(run['alpha'])
        num_videos = str(run['num_videos'])
        trace = run['training']
        generations = str(run['generations'])
        generation = str(run['generation'])
        
        print("Resuming", run_id, strategy, alpha, num_videos, f'{generation}/{generations}')
        proc = subprocess.Popen([sys.executable, 'main.py', 
                                 '--is-resume',
                                 '--run-id', run_id,
                                 '--strategy', strategy, 
                                 '--alpha', alpha, 
                                 '--num-videos', num_videos,
                                 '--generations', generations,
                                 '--initial-trace', trace,
                                 '--initial-generation', generation
                                 ])
        processes.append(proc)
        sleep(randint(1, 5))

    # wait for processes to finish
    print('waiting for processes to end')
    while len(processes) > 0:
        for process in processes:
            if process.poll() is not None:
                processes.remove(process)
        sleep(args.wait_time * 60)
