#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for i in 0 42; do
    python train.py --net CNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.003 --save;
done

for i in 0 42; do
    python train.py --net SNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.003 --save;
done

for i in 0 42; do
    python train.py --net STDP -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.003 --save;
done
