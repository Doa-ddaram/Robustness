#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for i in 0; do
    python main.py --net CNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-4;
    python main.py --net SNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-4;
    python main.py --net STDP -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-4;
done

for i in 0; do
    python main.py --net CNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-3;
    python main.py --net SNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-3;
    python main.py --net STDP -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-3;
done

for i in 0; do
    python main.py --net CNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-5;
    python main.py --net SNN -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-5;
    python main.py --net STDP -t 50 --seed $i --dset CIFAR10 --attack --epsilon 0.009 --save --learning_rate 1e-5;
done
