#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for i in 0; do
    python main.py --net STDP -t 100 --seed 0 --dset CIFAR10 --attack --epsilon 0.009 --learning_rate 1e-4 --batch_size 128 --num_worker 4 --timestep 10;
    python main.py --net SNN -t 100 --seed 0 --dset CIFAR10 --attack --epsilon 0.009 --learning_rate 1e-4 --batch_size 128 --num_worker 4 --timestep 10;
    python main.py --net CNN -t 100 --seed 0 --dset CIFAR10 --attack --epsilon 0.009 --learning_rate 1e-4 --batch_size 128 --num_worker 4;
done