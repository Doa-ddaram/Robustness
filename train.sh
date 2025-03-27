#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for i in 0 21 42 315; doz
    python train.py --net CNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1;
done

for i in 0 21 42 315; do
    python train.py --net SNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1;
done

for i in 0 21 42 315; do
    python train.py --net STDP -t 50 --seed $i --dset MNIST --attack --epsilon 0.1;
done




