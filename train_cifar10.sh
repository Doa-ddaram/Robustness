#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --net CNN -t 50 --seed 0 --dset CIFAR10 --attack --epsilon 0.018 --learning_rate 0.05 --batch_size 64 --timestep 20 --save;
python main.py --net SNN -t 50 --seed 0 --dset CIFAR10 --attack --epsilon 0.018 --learning_rate 0.05 --batch_size 64 --timestep 20 --save;
python main.py --net STDP -t 50 --seed 0 --dset CIFAR10 --attack --epsilon 0.018 --learning_rate 0.05 --batch_size 64 --timestep 20 --save;