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
        "--explainer", dest="explainer", help="Explaination methods."
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

    parser.set_defaults(
        mode="base10",
        explainer="IntegratedGradients",
        layer=12,
        label=0,
        batch_size=32,
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

netA = VGG_A(n_classes, layer=args.layer).to(device)
netB = VGG_B(n_classes, layer=args.layer).to(device)
netA.load_from_VGG(net.module)
netB.load_from_VGG(net.module)
netA = torch.nn.DataParallel(netA)
netB = torch.nn.DataParallel(netB)

if explanation_method == 'Saliency':
    explainer = Saliency(netB)
elif explanation_method == 'IntegratedGradients':
    explainer = IntegratedGradients(netB)
elif explanation_method == 'DeepLift':
    explainer = DeepLift(netB)
elif explanation_method == 'SmoothGrad':
    explainer = IntegratedGradients(netB)
elif explanation_method == "GradientSHAP":
    explainer = GradientShap(netB)
else:
    explainer = None

score = np.zeros((256, 1, 1))
print("Explanation method: ", explanation_method)
print("Number of samples: ", len(testloader)*batch_size)

start_time = time.time()
for inputs, labels in testloader:
    outputs = net(inputs.to(device))
    _, predicted = torch.max(outputs.data, 1)

    for ind in range(len(inputs)):
        img = inputs[ind].unsqueeze(0)
        img.requires_grad = True
        img = img.to(device)
        img = netA(img)

        if explanation_method == 'Saliency':
            grads = explainer.attribute(img, target=args.label)
            score = score + grads[0].cpu().detach().numpy()
        elif explanation_method == 'IntegratedGradients':
            attr_ig, delta = explainer.attribute(img, baselines=img * 0,
                                                 target=args.label,
                                                 return_convergence_delta=True)
            score = score + attr_ig[0].cpu().detach().numpy()
        elif explanation_method == 'DeepLift':
            attr_dl = explainer.attribute(img, baselines=img * 0,
                                          target=args.label)
            score = score + attr_dl[0].cpu().detach().numpy()
        elif explanation_method == 'SmoothGrad':
            noise_tunnel = NoiseTunnel(explainer)
            attributions_ig_nt = noise_tunnel.attribute(img, nt_type='smoothgrad_sq',
                                                        target=args.label)
            score = score + attributions_ig_nt[0].cpu().detach().numpy()
        elif explanation_method == "GradientSHAP":
            rand_img_dist = torch.cat([img * 0, img * 1])
            attributions_gs = explainer.attribute(img,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=args.label)
            score = score + attributions_gs[0].cpu().detach().numpy()
        else:
            score = score
all_duration = time.time() - start_time
print("Explaning duration: ", all_duration)

score = score / len(testloader) / batch_size

np.save(RESULT_DIR + args.explainer + '_' + str(mode) + '_' + str(args.label) + '_' + str(args.layer) + ".npy", score)

with open(RESULT_DIR + args.explainer + '_' + str(mode) + '_' + str(args.label) + '_' + str(args.layer) + ".txt", 'w') as f:
    f.write(str(all_duration) + "\n")
    f.write(str(score.shape))
