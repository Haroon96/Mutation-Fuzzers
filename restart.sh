#!/bin/bash
cd /mnt/data0/mharoon/Mutation-Fuzzers
tmux kill-session -t fuzzer
/home/mharoon/.conda/envs/fuzzer/bin/cleanfuzz
tmux new -d -s fuzzer
tmux send-keys -t fuzzer 'conda activate fuzzer' C-m
tmux send-keys -t fuzzer 'python resume.py -wt 5 -mp 50' C-m
