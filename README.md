Ready on KCC2025...

## train CNN(model : cnn, epochs : 10, seed : 0, dataset : MNIST or CIFAR10, attack status : True)
python -m train --net CNN -t 10 --seed 0 --dset MNIST --attack

## train SNN(model : snn, epochs : 10, seed : 0, dataset : MNIST or CIFAR10, attack status : False)
python -m train --net SNN -t 10 --seed 0 --dset MNIST --no-attack

## train STDP(model : snn_stdp, epochs : 10, seed : 0, dataset : MNIST or CIFAR10, attack status : True)
python -m train --net STDP -t 10 --seed 0 --dset MNIST --attack
