Ready on KCC2025...

## train CNN(model : cnn, epochs : 10, seed : 0, dataset : MNIST or CIFAR10)
python -m train --net CNN -t 10 --seed 0 --dset MNIST

## train SNN(model : snn, epochs : 10, seed : 0, dataset : MNIST or CIFAR10)
python -m train --net SNN -t 10 --seed 0 --dset MNIST

## train STDP(model : snn_stdp, epochs : 10, seed : 0, dataset : MNIST or CIFAR10)
python -m train --net STDP -t 10 --seed 0 --dset MNIST
