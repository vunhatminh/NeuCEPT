# NeuCEPT

Source code for the experiments in the paper `NeuCEPT: Locally Discover Neural Networks' Mechanism via Critical Neurons Identification with Precision Guarantee`. Our method is also named RATIONAL in some experiments.

## Directory structure

- `mnist`: source code for the LeNet model on MNIST
- `cifar10`: source code for the VGG11 model on CIFAR-10

## Dependencies
This codebase has been developed and tested only with python 3.8.10 and on a linux 64-bit operation system.

We recommend using `conda` to create a development environment with necessary packages. Refer to https://www.anaconda.com/products/individual for instructions on how to install `conda`.

### Set up a conda environment
We have prepared a file containing the same environment specifications that we use. To reproduce this environment (only on a linux 64-bit OS), execute the following command:

```bash
$ conda create --name rational --file spec-list.txt
```

## Usage
First, activate the created environment with the following command:

```bash
$ conda activate rational
```

Refer to the `README.md` files in the `mnist` and `cifar10` directories for more information on reproducing the experiments on the MNIST and CIFAR-10 datasets, respectively.
