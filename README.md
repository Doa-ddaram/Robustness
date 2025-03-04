Ready on KCC2025...

If adversial attack about model, input epsilon.

## train CNN(model : cnn, epochs : 10, seed : 0, dataset : MNIST or CIFAR10, attack status : True, epsilon : 0.1)
python -m train --net CNN -t 10 --seed 0 --dset MNIST --attack --epsilon 0.1

## train SNN(model : snn, epochs : 10, seed : 0, dataset : MNIST or CIFAR10, attack status : False)
python -m train --net SNN -t 10 --seed 0 --dset MNIST --no-attack

## train STDP(model : snn_stdp, epochs : 10, seed : 0, dataset : MNIST or CIFAR10, attack status : True, epsilon : 0.1)
python -m train --net STDP -t 10 --seed 0 --dset MNIST --attack
