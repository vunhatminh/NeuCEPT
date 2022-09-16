import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import LeNet

import time
import argparse

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GradientShap

def arg_parse():
    parser = argparse.ArgumentParser(description="MNIST class important neurons.")
    parser.add_argument(
            "--mode", dest="mode", help="Train mode: base, mod5 or mod2"
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
            "--layer", dest="layer", help="Layer to attribute ('linear0' or 'conv2' or 'conv3')."
        )
    
    parser.set_defaults(
        mode = "base",
        explainer = "Saliency",
        layer = 'conv3',
        label = 0,
        batch_size = 32,
    )
    return parser.parse_args()

prog_args = arg_parse()

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42

batch_size = prog_args.batch_size
mode = prog_args.mode
explanation_method = prog_args.explainer

IMG_SIZE = 32
N_CLASSES = 10

# define transforms
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

if mode == "mod2":
    train_dataset.targets[train_dataset.targets % 2 == 0] = 0
    train_dataset.targets[train_dataset.targets % 2 == 1] = 1
    valid_dataset.targets[valid_dataset.targets % 2 == 0] = 0
    valid_dataset.targets[valid_dataset.targets % 2 == 1] = 1
    N_CLASSES = 2
elif mode == "distillmod2":
    train_dataset.targets[train_dataset.targets % 2 == 0] = 0
    train_dataset.targets[train_dataset.targets % 2 == 1] = 1
    valid_dataset.targets[valid_dataset.targets % 2 == 0] = 0
    valid_dataset.targets[valid_dataset.targets % 2 == 1] = 1
    N_CLASSES = 2
elif mode == "mod5":
    train_dataset.targets[train_dataset.targets % 5 == 0] = 0
    train_dataset.targets[train_dataset.targets % 5 != 0] = 1
    valid_dataset.targets[valid_dataset.targets % 5 == 0] = 0
    valid_dataset.targets[valid_dataset.targets % 5 != 0] = 1
    N_CLASSES = 2
elif mode == "distillmod5":
    train_dataset.targets[train_dataset.targets % 5 == 0] = 0
    train_dataset.targets[train_dataset.targets % 5 != 0] = 1
    valid_dataset.targets[valid_dataset.targets % 5 == 0] = 0
    valid_dataset.targets[valid_dataset.targets % 5 != 0] = 1
    N_CLASSES = 2

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=batch_size, 
                          shuffle=False)

torch.manual_seed(RANDOM_SEED)

net = LeNet.LeNet5(N_CLASSES).to(DEVICE)

LOAD_PATH = 'pretrained/mnist'
if mode == "mod2":
    LOAD_NAME = LOAD_PATH + '_mod2'
elif mode == "distillmod2":
    LOAD_NAME = LOAD_PATH + '_distillmod2'
elif mode == "mod5":
    LOAD_NAME = LOAD_PATH + '_mod5'
elif mode == "distillmod5":
    LOAD_NAME = LOAD_PATH + '_distillmod5'
else:
    LOAD_NAME = LOAD_PATH + '_base'
print("Using existing trained model at " + LOAD_NAME)

net.load_state_dict(torch.load(LOAD_NAME))
net.eval()

train_acc = LeNet.get_accuracy(net, train_loader, device=DEVICE)
valid_acc = LeNet.get_accuracy(net, valid_loader, device=DEVICE)
print(f'Train accuracy: {100 * train_acc:.2f}\t'
      f'Valid accuracy: {100 * valid_acc:.2f}')

netA = LeNet.LeNet5_A(prog_args.layer).to(DEVICE)
netB = LeNet.LeNet5_B(prog_args.layer, N_CLASSES).to(DEVICE)
netA.load_from_LeNet(net)
netB.load_from_LeNet(net)

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

if prog_args.layer == 'conv3':
    layer_shape = 120
elif prog_args.layer == 'conv2':
    layer_shape = (16,10,10)
elif prog_args.layer == 'linear0':
    layer_shape = 84

score = np.zeros(layer_shape)
print("Explanation method: ", explanation_method)
print("Number of samples: ", len(valid_loader)*batch_size)
start_time = time.time()
for inputs, labels in valid_loader:
    outputs = net(inputs.to(DEVICE))
    _, predicted = torch.max(outputs.data, 1)

    for ind in range(len(inputs)):
        img = inputs[ind].unsqueeze(0)
        img.requires_grad = True
        img = img.to(DEVICE)
        img = netA(img)

        if explanation_method == 'Saliency':
            grads = explainer.attribute(img, target=prog_args.label)
            score = score + grads.squeeze().cpu().detach().numpy()
        elif explanation_method == 'IntegratedGradients':
            attr_ig, delta = explainer.attribute(img, baselines=img * 0,
                                                target=prog_args.label,
                                                return_convergence_delta=True)
            score = score + attr_ig.squeeze().cpu().detach().numpy()
        elif explanation_method == 'DeepLift':
            attr_dl = explainer.attribute(img, baselines = img * 0,
                                        target=prog_args.label)
            score = score + attr_dl.squeeze().cpu().detach().numpy()
        elif explanation_method == 'SmoothGrad':
            noise_tunnel = NoiseTunnel(explainer)
            attributions_ig_nt = noise_tunnel.attribute(img, nt_type='smoothgrad_sq', 
                                                        target=prog_args.label)
            score = score + attributions_ig_nt.squeeze().cpu().detach().numpy()
        elif explanation_method == "GradientSHAP":
            rand_img_dist = torch.cat([img * 0, img * 1])
            attributions_gs = explainer.attribute(img,
                                                n_samples=50,
                                                stdevs=0.0001,
                                                baselines=rand_img_dist,
                                                target=prog_args.label)
            score = score + attributions_gs.squeeze().cpu().detach().numpy()
        else:
            score = score
all_duration = time.time() - start_time
start_time = time.time()
print("Explaning duration: ", all_duration)
    
score = score/len(valid_loader)/batch_size

def gen_filesave(args):
    if args.mode == 'base':
        name = 'result/'+ args.explainer + '_' + str(args.label) + '_' + args.layer + ".npy"
    else:
        name = 'result/'+ args.mode + "_" + args.explainer + '_' + str(args.label) + '_' + args.layer + ".npy"
    return name

savename = gen_filesave(prog_args)
np.save(savename, score)

def gen_logfile(args):
    if args.mode == 'base':
        name = 'result/'+ args.explainer + '_' + str(args.label) + '_' + args.layer + ".txt"
    else:
        name = 'result/'+ args.mode + "_" + args.explainer + '_' + str(args.label) + '_' + args.layer + ".txt"
    return name

logname = gen_logfile(prog_args)
with open(logname, 'w') as f:
    f.write(str(all_duration) + "\n")
    
