#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for i in 0; do
    python train.py --net CNN -t 0 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-4 --indicate;
    python train.py --net SNN -t 0 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-4 --indicate;
    python train.py --net STDP -t 0 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-4 --indicate;
done

for i in 0; do
    python train.py --net CNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-3 --indicate;
    python train.py --net SNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-3 --indicate;
    python train.py --net STDP -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-3 --indicate;
done

for i in 0; do
    python train.py --net CNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-5 --indicate;
    python train.py --net SNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-5 --indicate;
    python train.py --net STDP -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-5 --indicate;
done
