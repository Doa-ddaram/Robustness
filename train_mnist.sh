#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for i in 0 42; do
    python train.py --net CNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-2 --indicate;
    python train.py --net SNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-2 --indicate;
    python train.py --net STDP -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-2 --indicate;
done

for i in 0 42; do
    python train.py --net CNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-3 --indicate;
    python train.py --net SNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-3 --indicate;
    python train.py --net STDP -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-3 --indicate;
done

for i in 0 42; do
    python train.py --net CNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-4 --indicate;
    python train.py --net SNN -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-4 --indicate;
    python train.py --net STDP -t 50 --seed $i --dset MNIST --attack --epsilon 0.1 --save --learning_rate 1e-4 --indicate;
done