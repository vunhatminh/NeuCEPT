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
              "--explainer",  
            nargs="*",  
            default=["FDR"],  
            help="Explaination methods."
            )
    parser.add_argument(
            "--label", dest="label", type=int, help="Class of interest."
        )
    parser.add_argument(
            "--top", dest="top", type=int, help="Number of top neurons."
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
    parser.add_argument(
            "--batchsize", dest="batch_size", type=int, help="Batch size."
        )
    parser.add_argument(
            "--layer", dest="layer", help="Layer to attribute ('linear0' or 'conv3')."
        )
    
    parser.set_defaults(
        mode = "base",
        explainer = ["FDR"],
        label = 0,
        top = None,
        noise_max = 50.0,
        noise_step = 10.0,
        gamma = 10.0,
        batch_size = 32,
        layer = 'linear0'
    )
    return parser.parse_args()

prog_args = arg_parse()

print("ABALATION TEST FOR LAYER ", prog_args.layer)

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
IMG_SIZE = 32
N_CLASSES = 10

batch_size = prog_args.batch_size
mode = prog_args.mode

# define transforms
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

train_dataset.targets = train_dataset.targets.clone().detach()
idx = train_dataset.targets == prog_args.label
train_dataset.targets= train_dataset.targets[idx]
train_dataset.data = train_dataset.data[idx.numpy().astype(bool)]
print("Number of samples in train set: ", len(train_dataset))

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)
valid_dataset.targets = valid_dataset.targets.clone().detach()
idx = valid_dataset.targets == prog_args.label
valid_dataset.targets= valid_dataset.targets[idx]
valid_dataset.data = valid_dataset.data[idx.numpy().astype(bool)]
print("Number of samples in test set: ", len(valid_dataset))

if mode == "mod2":
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
elif mode == "mod5":
    LOAD_NAME = LOAD_PATH + '_mod5'
else:
    LOAD_NAME = LOAD_PATH + '_base'
print("Using existing trained model at " + LOAD_NAME )

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

def gen_filesaves(args):
    names = []
    for method in args.explainer:
        if args.layer == None:
            name = 'result/'+ method + '_' + str(args.label) + ".npy"
        else:
            name = 'result/'+ method + '_' + str(args.label) + '_' + args.layer + ".npy"
        names.append(name)
    return names

def gen_panda_save(args):
    if args.layer == None:
        if args.top == None: 
            name = 'result/Ablation_Label_' + str(args.label) + "_Gamma_" + str(args.gamma) + ".pkl"
        else:
            name = 'result/Ablation_Label_' + str(args.label) + "_Gamma_" + str(args.gamma) + "_Top_" + str(args.top) +".pkl"
    else:
        if args.top == None: 
            name = 'result/Ablation_Layer ' + args.layer + '_Label_' + str(args.label) + "_Gamma_" + str(args.gamma) + ".pkl"
        else:
            name = 'result/Ablation_Layer ' + args.layer + '_Label_' + str(args.label) + "_Gamma_" + str(args.gamma) + "_Top_" + str(args.top) +".pkl"
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
            
            if args.layer == None:
                weighted_noise = base_noise[sample][0]*weighted_noise
            elif args.layer == 'linear0':
                weighted_noise = base_noise[sample]*weighted_noise
            elif args.layer == 'conv3':
                weighted_noise = base_noise[sample]*weighted_noise
            weighted_noise = weighted_noise / np.sqrt(np.sum(weighted_noise**2)) * np.sqrt(noise_)
            
            if args.layer == None:
                perturb_dict[method][sample][0] = perturb_dict[method][sample][0] + weighted_noise
            elif args.layer == 'linear0':
                perturb_dict[method][sample] = perturb_dict[method][sample] + weighted_noise
            elif args.layer == 'conv3':
                perturb_dict[method][sample] = perturb_dict[method][sample] + weighted_noise
                
    return perturb_dict

                                                     
    
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

ablation_accuracy = pd.DataFrame()
noise_levels = list(np.array(range(int(prog_args.noise_max/prog_args.noise_step)))*prog_args.noise_step)
for noise in noise_levels:
    corrects = get_perturb_accuracy(prog_args, netA, netB, train_loader, DEVICE,
                             noise = noise, gamma = prog_args.gamma)
    ablation_accuracy = ablation_accuracy.append(corrects, ignore_index=True)
    print("NOISE LEVEL: ", noise)
    print(corrects)

ablation_accuracy['Noise_level'] = noise_levels
ablation_accuracy.to_pickle(gen_panda_save(prog_args))