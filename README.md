# STDP-based Training for Adversarially Robust Spiking Neural Networks

This repository Contains DSR training code.

## Overview

Spiking Neural Networks (SNNs) have gained attention due to their energy efficiency and event-driven nature.

## 🧪 Key Findings

| Dataset  | Model       | Test Acc (%) | Adv. Acc (%) | ASR (%)   |
|----------|-------------|--------------|--------------|-----------|
| CIFAR-10 | CNN         | 0.0          | 0.0          | 0.0       |
| CIFAR-10 | SNN         | 0.0          | 0.0          | 0.0       |
| CIFAR-10 | SNN + STDP  | 0.0          | 0.0          | 0.0       |

## ⚙️ Repository Structure

```bash
Robustness/
├── requirements.txt        
├── modules/                # Model definitions (SNN, DSR-SNN)
├── imagenet/               # ImageNet-1K training and evaluation code
│   └── main.py             
├── cifar/                  # CIFAR-10 training and evaluation code
│    └── main.py             
├── train_cifar10.sh        # Training script for CIFAR-10 dataset
├── train_imagenet1k.sh     # Training script for ImageNet-1K dataset
├── visualizing.py          # Visualization of spikes and weights
└── README.md               # Project documentation
```

## How to run?

#### Train Dataset : CIFAR10

```bash
python -m cifar.main --path ./data --dataset cifar10 --model [model_name] --name [checkpoint_name]
```

#### Train Dataset : ImageNet-1K

```bash
python -m imagenet.main --path ./data --dataset imagenet1k --model [model_name] --name [checkpoint_name]
```

## 📌 Notes
* This branch focuses on DSR-based spike representation.

* STDP integration is optional but improves sparsity and robustness.