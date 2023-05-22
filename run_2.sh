#!/bin/bash

cd hw4/meta_rl/
source /home/paulemile/anaconda3/bin/activate cs224r-meta-rl
python scripts/dream.py dream_gpu_vm -b environment=\"map\" --force_overwrite
