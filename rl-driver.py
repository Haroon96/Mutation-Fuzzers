import subprocess
from argparse import ArgumentParser
import sys
from time import sleep
from random import randint
import os
from natsort import natsorted

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-mp', '--max-parallel', type=int, default=5)
    parser.add_argument('-wt', '--wait-time', type=float, default=5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    processes = []

    strategy = 'rl'
    weight_dir = 'rl_fuzzer/saved_weights'
    weights = natsorted(os.listdir(weight_dir))
    while True:
        for w in weights:
            if not w.startswith('actor'):
                continue
            
            # sleep if max processes
            while len(processes) >= args.max_parallel:
                sleep(args.wait_time * 60)
                
                # find and remove finished processes
                for process in processes:
                    if process.poll() is not None:
                        processes.remove(process)
                    
            # spawn process
            print("Starting", strategy, w)
            proc = subprocess.Popen([sys.executable, 'concurrent-main.py', '--strategy', strategy, '--strategy-params', os.path.join(weight_dir, w), '--num-videos', '30', '--generations', '50', '--alpha', '0.5'])
            processes.append(proc)
            sleep(randint(1, 15))
