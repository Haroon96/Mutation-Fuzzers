import subprocess
from argparse import ArgumentParser
import sys
from time import sleep
from random import randint

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-mp', '--max-parallel', type=int, default=5)
    parser.add_argument('-wt', '--wait-time', type=float, default=5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    processes = []

    while True:
        for alpha in ['0.1', '0.2', '0.3', '0.5']:
            for strategy in ['greedy', 'random']:
                
                # sleep if max processes
                while len(processes) >= args.max_parallel:
                    sleep(args.wait_time * 60)
                    
                    # find and remove finished processes
                    for process in processes:
                        if process.poll() is not None:
                            processes.remove(process)
                        
                # spawn process
                print("Starting", strategy, alpha)
                proc = subprocess.Popen([sys.executable, 'main.py', '--strategy', strategy, '--alpha', alpha, '--num-videos', '30'])
                processes.append(proc)
                sleep(randint(1, 5))
