#!/bin/sh

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130.1 cudnn/7.6/7.6.2
module load python/3.6/3.6.5
source ~/venv/pytransfer/bin/activate


# MNISTR
test_domain_list="M0 M15 M30 M45 M60 M75"
reg_weight_list="0.0 0.1 0.2 0.5 1.0 2.0 5.0"

for seed in $1
do
    for test_domain in $test_domain_list
    do
        for reg_weight in $reg_weight_list
        do
            python3 ~/pytransfer/examples/dg_dan.py --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed
            python3 ~/pytransfer/examples/dg_dan.py --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed --reg_name dan-ent
            python3 ~/pytransfer/examples/dg_dan.py --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed --reg_name ensemble
        done
    done
done

# OppG OppL
test_domain_list="S1 S2 S3 S4"
reg_weight_list="0.0 0.01 0.1 0.2 0.5 1.0 2.0 5.0"

for seed in $1
do
    for test_domain in $test_domain_list
    do
        for reg_weight in $reg_weight_list
        do
            python3 ~/pytransfer/examples/dg_dan.py --dataset_name oppG --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed
            python3 ~/pytransfer/examples/dg_dan.py --dataset_name oppG --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed --reg_name dan-ent
            python3 ~/pytransfer/examples/dg_dan.py --dataset_name oppG --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed --reg_name ensemble
            
            python3 ~/pytransfer/examples/dg_dan.py --dataset_name oppL --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed
            python3 ~/pytransfer/examples/dg_dan.py --dataset_name oppL --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed --reg_name dan-ent
            python3 ~/pytransfer/examples/dg_dan.py --dataset_name oppL --epoch 100 --reg_weight $reg_weight --test_domain $test_domain --seed $seed --reg_name ensemble
        done
    done
done
deactivate
