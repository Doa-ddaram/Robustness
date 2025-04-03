#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for i in 0; do
    python main.py --net CNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.2 --learning_rate 1e-2 --indicate;
    python main.py --net SNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.2 --learning_rate 1e-2 --indicate; 
    python main.py --net STDP  -t 50 --seed $i --dset MNIST --attack --epsilon 0.2 --learning_rate 1e-2 --indicate;
done

