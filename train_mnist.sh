#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --net CNN -t 50 --seed 50 --dset MNIST --attack --epsilon 0.2 --learning_rate 1e-3 --load;
python main.py --net SNN -t 50 --seed 50 --dset MNIST --attack --epsilon 0.2 --learning_rate 1e-3 --timestep 10 --load;
python main.py --net STDP -t 50 --seed 50 --dset MNIST --attack --epsilon 0.2 --learning_rate 1e-3 --timestep 10 --load;
