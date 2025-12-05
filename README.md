# Deep Learning for Single-Cell RNA-seq Analysis

A comparison of deep learning approaches for analyzing single-cell gene expression data. 

## Overview

This project explores different neural network architectures for single-cell RNA sequencing (scRNA-seq) data, comparing their performance across various embedding strategies. 

## Models

- **MLP** — with PCA, scGPT, and Gene-Aware embeddings
- **VAE** — Variational Autoencoder for dimensionality reduction
- **LSTM** — Sequential modeling of gene expression
- **Transformer** — Attention-based architecture

## Project Structure

```
├── scripts/          # Core modules (data loading, models, training, utils)
├── train_scripts/    # Training scripts for each model variant
└── Results/          # Saved model outputs and metrics
```

## Results

Trained models and evaluation metrics are saved in the `Results/` folder, organized by architecture. 

## Notebook

The notebook reproduces the main results, importing functions from our main scripts.
