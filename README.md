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
├── images/                 # Visualization results (e.g., spike raster, weight sparsity)
├── utils/                  # Dataset loading, attack methods (FGSM, PGD), training utilities
├── modules/                # Model definitions (CNN, SNN, DSR-SNN)
├── train_cifar10.sh        # Training script for CIFAR10 dataset
├── train_mnist.sh          # Training script for MNIST dataset
├── main.py                 # Main training and evaluation entry point
├── dsr_trainer.py          # DSR-based training pipeline
├── stdp_trainer.py         # STDP-based unsupervised training pipeline
├── visualizing.py          # Visualization of spikes and weights
├── requirements.txt        # Python dependencies
└── README.md
```

## How to run?

Implementation is planned soon.

## 📌 Notes
* This branch focuses on DSR-based spike representation.

* STDP integration is optional but improves sparsity and robustness.