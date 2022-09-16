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
        "--explainer", dest="explainer", help="Explaination methods.", nargs="*"
    )
    parser.add_argument(
        "--top", dest="top", type=int, help="Number of top neurons."
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
        "--noise_max", dest="noise_max", type=float
    )
    parser.add_argument(
        "--noise_step", dest="noise_step", type=float
    )
    parser.add_argument(
        "--gamma", dest="gamma", type=float, help="Gamma value for weighted noise generation."
    )

    parser.set_defaults(
        mode = "base10",
        explainer = ["Saliency"],
        label = 0,
        top = None,
        noise_max = 50.0,
        noise_step = 10.0,
        gamma = 10.0,
        batch_size = 32,
        layer = 12
    )
    return parser.parse_args()


args = arg_parse()

explanation_method = args.explainer
batch_size = args.batch_size
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
mode = args.mode

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
    path = PRETRAINED_DIR + 'cifar-distill.pth'
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

netA = VGG_A(n_classes, layer=args.layer).to(device)
netB = VGG_B(n_classes, layer=args.layer).to(device)
netA.load_from_VGG(net.module)
netB.load_from_VGG(net.module)
netA = torch.nn.DataParallel(netA)
netB = torch.nn.DataParallel(netB)


def gen_filesaves(args):
    names = []
    for method in args.explainer:
        name = RESULT_DIR + method + '_' + str(args.label) + '_' + str(args.layer) + ".npy"
        names.append(name)
    return names


def gen_panda_save(args):
    name = 'result/Ablation_Layer ' + str(args.layer) + '_Label_' + str(args.label) + "_Gamma_" + str(args.gamma) + "_Top_" + str(args.top) +".pkl"
    return name


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


from PIL import Image


def load_score(args):
    load_files = gen_filesaves(args)
    score_list = []
    mask_list = []
    for f in load_files:
        neuron_score = np.load(f)
        
        # For FDR only
        neuron_score = neuron_score.reshape((256,1,1))

        if args.layer == None: # if layer is the input, convert to input dimension
            PIL_image = Image.fromarray(neuron_score[0])
            PIL_image = PIL_image.resize((32,32), Image.NEAREST)
            neuron_score = np.array(PIL_image)

        if args.top != None: # if top is specified, return 0-1 mask with top k neurons
            mask = np.zeros_like(neuron_score)
            mask[largest_indices(neuron_score, args.top)] = 1
            neuron_score = mask

        # Normalized
        neuron_score = (neuron_score - np.min(neuron_score))/(np.max(neuron_score) - np.min(neuron_score))
        neuron_score = neuron_score / np.sqrt(np.sum(neuron_score**2))
        score_list.append(neuron_score)

    score_dict = dict(zip(args.explainer, score_list))
    return score_dict


def gen_perturb(args, data, data_max = None, data_min = None, noise = None, gamma = None):
    score_dict = load_score(args)

    if data_max == None:
        d_max = torch.max(data).item()
    else:
        d_max = data_max
    if data_min == None:
        d_min = torch.min(data).item()
    else:
        d_min = data_min

    if noise == None:
        noise_ = args.noise
    else:
        noise_ = noise
    if gamma == None:
        gamma_ = args.gamma
    else:
        gamma_ = gamma


    perturb_dict = {}
    for method in args.explainer:
        perturb_dict[method] = data.clone()

    for sample in range(data.shape[0]):
        base_noise = np.random.rand(*data.shape)
        for method in args.explainer:
            weighted_noise = score_dict[method]*gamma_
            weighted_noise = 2**(-weighted_noise)

            weighted_noise = base_noise[sample]*weighted_noise
            weighted_noise = weighted_noise / np.sqrt(np.sum(weighted_noise**2)) * np.sqrt(noise_)

            perturb_dict[method][sample] = perturb_dict[method][sample] + weighted_noise

    return perturb_dict


def get_perturb_accuracy(args, modelA, modelB, data_loader, device,
                         noise = None, gamma = None):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    if noise == None:
        noise_ = args.noise
    else:
        noise_ = noise
    if gamma == None:
        gamma_ = args.gamma
    else:
        gamma_ = gamma

    correct_pred = 0
    n = 0
    corrects = dict(zip(args.explainer , list(np.zeros(len(args.explainer)))))
    with torch.no_grad():
        modelA.eval()
        modelB.eval()
        for X, y_true in data_loader:
            y_true = y_true.to(device)
            X_A = modelA(X.to(device))
            perturbs = gen_perturb(args, X_A.cpu(),
                                   noise = noise_,
                                   gamma = gamma_)
            for method in args.explainer:
                perturbs[method] = perturbs[method].to(device)
                y_hat = modelB(perturbs[method])
                y_prob = F.softmax(y_hat, dim=1)
                _, predicted_labels = torch.max(y_prob, 1)
                corrects[method] += (predicted_labels == y_true).sum().item()
            n += y_true.size(0)

    for method in args.explainer:
        corrects[method] = corrects[method]/n

    return corrects


import pandas as pd
ablation_accuracy = pd.DataFrame()
noise_levels = list(np.array(range(int(args.noise_max/args.noise_step)))*args.noise_step)
for noise in noise_levels:
    corrects = get_perturb_accuracy(args, netA, netB, trainloader, device,
                                    noise=noise, gamma=args.gamma)
    ablation_accuracy = ablation_accuracy.append(corrects, ignore_index=True)
    print("NOISE LEVEL: ", noise)
    print(corrects)

ablation_accuracy['Noise_level'] = noise_levels
ablation_accuracy.to_pickle(gen_panda_save(args))