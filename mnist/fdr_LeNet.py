import numpy as np
from datetime import datetime 
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import LeNet
from prober import Prober
import pandas as pd
import knockpy
from knockpy.knockoff_filter import KnockoffFilter

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
            "--fdr", dest="fdr", type=float, help="FDR."
        )
    parser.add_argument(
            "--runs", dest="runs", type=int, help="Number of runs."
        )
    parser.add_argument(
            "--batchsize", dest="batch_size", type=int, help="Batch size."
        )
    parser.add_argument(
            "--layer", dest="layer", help="Layer to attribute (None or 'linear0' or 'conv3')."
        )
    
    parser.set_defaults(
        mode = "base",
        explainer = "FDR",
        label = 0,
        fdr = 0.1,
        runs = 1,
        batch_size = 32,
        layer = None
    )
    return parser.parse_args()

prog_args = arg_parse()

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
IMG_SIZE = 32
N_CLASSES = 10

batch_size = prog_args.batch_size
mode = prog_args.mode
explanation_method = prog_args.explainer
mode = prog_args.mode
examined_layer = prog_args.layer
fdr = prog_args.fdr
runs = prog_args.runs

if examined_layer == None:
    print(explanation_method + " for Input layer")
else:
    print(explanation_method + " for " + examined_layer)

# define transforms
transforms_ = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms_,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms_)

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
print("Using existing trained model at " + LOAD_NAME )

net.load_state_dict(torch.load(LOAD_NAME))
net.eval()

train_acc = LeNet.get_accuracy(net, train_loader, device=DEVICE)
valid_acc = LeNet.get_accuracy(net, valid_loader, device=DEVICE)
print(f'Train accuracy: {100 * train_acc:.2f}\t'
      f'Valid accuracy: {100 * valid_acc:.2f}')


probing = Prober(net)
probing.compute_dataset_activation(valid_dataset, device = DEVICE)
activation = probing.activation

if examined_layer == 'conv3':
    layer_shape = 120
elif examined_layer == 'linear0':
    layer_shape = 84
else:
    layer_shape = (8,8)

start_time = time.time()

score = np.zeros(layer_shape)

if examined_layer == 'linear0':
    X_var = np.array(probing.activation['classifier_1'].view(len(valid_dataset),-1).cpu())
elif examined_layer == 'conv3':
    X_var = np.array(probing.activation['feature_extractor_7'].view(len(valid_dataset),-1).cpu())
else:
    pre_process = transforms.Compose([transforms.Resize((8,8))])
    X_var = np.array(pre_process(probing.activation['in']).view(len(valid_dataset),-1).cpu())

y_var = np.array(probing.activation['out'][:,prog_args.label].cpu())

for _ in range(runs):
    kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
    rejections = kfilter.forward(X=X_var, y=y_var, fdr=fdr)
    top_neurons = np.where(rejections==1)[0]
    explanation = np.zeros(X_var.shape[1])
    explanation[top_neurons] = 1
    explanation = explanation.reshape(layer_shape)
    score = score + explanation   

all_duration = time.time() - start_time
start_time = time.time()
print("Explaning duration: ", all_duration)
    
score = score/len(valid_loader)/batch_size

def gen_filesave(args):
    if args.mode == 'base':
        if args.layer == None:
            name = 'result/'+ args.explainer + '_' + str(args.label) + ".npy"
        else:
            name = 'result/'+ args.explainer + '_' + str(args.label) + '_' + args.layer + ".npy"
    else:
        if args.layer == None:
            name = 'result/'+ args.mode + "_" + args.explainer + '_' + str(args.label) + ".npy"
        else:
            name = 'result/'+ args.mode + "_" + args.explainer + '_' + str(args.label) + '_' + args.layer + ".npy"
    return name

savename = gen_filesave(prog_args)
np.save(savename, score)

def gen_logfile(args):
    if args.mode == 'base':
        if args.layer == None:
            name = 'result/'+ args.explainer + '_' + str(args.label) + ".txt"
        else:
            name = 'result/'+ args.explainer + '_' + str(args.label) + '_' + args.layer + ".txt"
    else:
        if args.layer == None:
            name = 'result/'+ args.mode + "_" + args.explainer + '_' + str(args.label) + ".txt"
        else:
            name = 'result/'+ args.mode + "_" + args.explainer + '_' + str(args.label) + '_' + args.layer + ".txt"
    return name

logname = gen_logfile(prog_args)
with open(logname, 'w') as f:
    f.write("Duration: " + str(all_duration) + "\n")
    if prog_args.layer == None:
        ratio = np.sum(score/np.max(score))/64
    else:
        ratio = np.sum(score/np.max(score))/layer_shape
    f.write("Selected: " + str(ratio) + "\n")
