# STDP-based Training for Adversarially Robust Spiking Neural Networks

This repository contains the code implementation of the paper:

> **스파이킹 신경망의 적대적 강건성을 위한 STDP 기반 학습**  
> 이원모, 성백륜, 고상기  
> University of Seoul, Department of AI, CIDA Lab

## 🧠 Overview

Spiking Neural Networks (SNNs) are attracting attention due to their energy efficiency and biological plausibility. However, their vulnerability to adversarial attacks remains a challenge.

This work explores how **STDP (Spike-Timing Dependent Plasticity)**, a biologically inspired unsupervised learning rule, can be used to improve the **adversarial robustness** of SNNs. The study compares:

- Conventional CNNs
- Surrogate gradient-based SNNs
- STDP-enhanced SNNs

against adversarial attacks such as FGSM on MNIST and CIFAR-10 datasets.

## 🧪 Key Findings

| Dataset  | Model       | Test Acc (%) | Adv. Acc (%) | ASR (%)   |
|----------|-------------|--------------|--------------|-----------|
| MNIST    | CNN         | 99.24        | 41.67        | 57.57     |
| MNIST    | SNN         | 99.20        | 70.90        | 28.27     |
| MNIST    | SNN + STDP  | 99.25        | 80.57        | 18.78     |
| CIFAR-10 | CNN         | 87.09        | 50.95        | 37.31     |
| CIFAR-10 | SNN         | 75.39        | 65.05        | 11.82     |
| CIFAR-10 | SNN + STDP  | 75.05        | 64.32        | 10.71     |

- **SNN+STDP** models showed significantly lower adversarial attack success rates compared to CNNs and vanilla SNNs.
- STDP encourages sparse connectivity, which limits gradient-based attack pathways.

## ⚙️ Repository Structure

```bash
Robustness/
├── images/               # Visualization results (e.g., weight histogram)
├── utils/                # Dataset loading, attack methods, and training utilities
├── train_cifar10.sh      # Main training script on CIFAR10 dataset
├── train_mnist.sh        # Main training script on MNIST dataset
├── main.py               # Main training and evaluation entry point
├── visualizing.py        # Visualization of learned weight sparsity
├── requirements.txt      # Python dependencies
└── README.md
```

## How to run?

### Run all models (with shell scripts)
 Train all three models (CNN, SNN, STDP) at once.

#### Dataset : MNIST
```bash
sh train_mnist.sh
```

#### Dataset : CIFAR10

```bash
sh train_cifar10.sh
```

### Train individual models (manual)
 Train individual models selectively.
#### Train CNN
```bash
python -m train --net CNN -t 10 --seed 0 --dset MNIST --attack --epsilon 0.1
```

#### Train SNN
```bash
python -m train --net SNN -t 10 --seed 0 --dset MNIST --attack --epsilon 0.1
```

#### Train STDP
```bash
python -m train --net STDP -t 10 --seed 0 --dset MNIST --attack --epsilon 0.1
```
