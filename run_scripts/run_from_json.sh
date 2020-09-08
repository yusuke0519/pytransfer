#!/bin/sh

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130.1 cudnn/7.6/7.6.2
module load python/3.6/3.6.5
source ~/venv/pytransfer/bin/activate


python3 ~/pytransfer/examples/dg_dan.py json -f './jsons/0045b85b-ae5c-4409-83e5-70e6c1c5e436.json'
