import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision
import argparse
import os

import time

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GradientShap

from models.vgg import *
from prober import Prober
# from utils import progress_bar

PRETRAINED_DIR = './pretrained/'
RESULT_DIR = './result/'
assert os.path.isdir(PRETRAINED_DIR), 'Error: no pretrained directory found!'
assert os.path.isdir(RESULT_DIR), 'Error: no result directory found!'


def arg_parse():
    parser = argparse.ArgumentParser(description="CIFAR10 class important neurons.")
    parser.add_argument(
        "--mode", dest="mode", help="Train mode: base10, base2 or distill"
    )
    parser.add_argument(
        "--label", dest="label", type=int, help="Class of interest."
    )
    parser.add_argument(
        "--batchsize", dest="batch_size", type=int, help="Batch size."
    )
    parser.add_argument(
        "--layer", dest="layer", type=int, help="Layer to attribute (12 or 13)."
    )
    parser.add_argument(
        "--fdr", dest="fdr", type=float, help="FDR."
    )
    parser.add_argument(
        "--runs", dest="runs", type=int, help="Number of runs."
    )

    parser.set_defaults(
        mode="base10",
        layer=12,
        label=0,
        batch_size=32,
        fdr = 0.6,
        runs = 1,
    )
    return parser.parse_args()


args = arg_parse()

batch_size = args.batch_size
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
mode = args.mode
examined_layer = args.layer
fdr = args.fdr
runs = args.runs


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

animals = [2,3,4,5,6,7]
not_animals = [0,1,8,9]

if mode != "base10":
    trainset.targets = torch.LongTensor(trainset.targets)
    for target in not_animals:
        trainset.targets[trainset.targets == target] = 0
    for target in animals:
        trainset.targets[trainset.targets == target] = 1
    testset.targets = torch.LongTensor(testset.targets)
    for target in not_animals:
        testset.targets[testset.targets == target] = 0
    for target in animals:
        testset.targets[testset.targets == target] = 1


if mode == "base10":
    path = PRETRAINED_DIR + 'cifar-base10.pth'
elif mode == "base2":
    path = PRETRAINED_DIR + 'cifar-base2.pth'
elif mode == 'distill':
    path = PRETRAINED_DIR + 'cifar-distill-freeze13.pth'
print('==> Loading pretrained model at ' + path)

if mode == "base10":
    n_classes = 10
    net = VGG(10, 'VGG11')
elif mode == "base2" or mode == 'distill':
    n_classes = 2
    net = VGG(2, 'VGG11')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

saved = torch.load(path)
net.load_state_dict(saved['net'])
print('Accuracy: ', saved['acc'])

start_time = time.time()
probing = Prober(net.module)
probing.compute_dataset_activation(testset, device = device)
activation = probing.activation

score = np.zeros(256)

if examined_layer == 12:
    X_var = np.array(probing.activation['features_29'].view(len(testset),-1).cpu())
elif examined_layer == 13:
    X_var = np.array(probing.activation['features_32'].view(len(testset),-1).cpu())
    if mode == 'base2' or mode == 'distill':
        X_var = X_var + np.random.normal(0, 0.00001, X_var.shape)  # matrix is not positive definite if not do this

y_var = np.array(probing.activation['out'][:,args.label].cpu())


for _ in range(runs):
    from knockpy.knockoff_filter import KnockoffFilter
    kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
    rejections = kfilter.forward(X=X_var, y=y_var, fdr=fdr)
    top_neurons = np.where(rejections == 1)[0]
    explanation = np.zeros(X_var.shape[1])
    explanation[top_neurons] = 1
    explanation = explanation.reshape(256)
    score = score + explanation

score = score.reshape((256,1,1))
all_duration = time.time() - start_time

print("Explaning duration: ", all_duration)
np.save(RESULT_DIR + "FDR" + '_' + str(mode) + '_' + str(args.label) + '_' + str(args.layer) + ".npy", score)

with open(RESULT_DIR + "FDR" + '_' + str(mode) + '_' + str(args.label) + '_' + str(args.layer) + ".txt", 'w') as f:
    f.write(str(all_duration) + "\n")
    f.write(str(score.shape) + '\n')
    ratio = np.sum(score/np.max(score))/256
    f.write("Selected: " + str(ratio) + "\n")
