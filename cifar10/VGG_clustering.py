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

from prober import Prober
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn import cluster
from sklearn import mixture

from models.vgg import *
# from utils import progress_bar

PRETRAINED_DIR = './pretrained/'
RESULT_DIR = './result/'
assert os.path.isdir(PRETRAINED_DIR), 'Error: no pretrained directory found!'
assert os.path.isdir(RESULT_DIR), 'Error: no result directory found!'


def gen_panda_save(args, cluster_method):
    if args.layer == None:
        name = 'result/' + args.mode + "_ClusMethod_" + str(cluster_method) + "_top_" + str(args.top) + ".pkl"
    else:
        name = 'result/' + args.mode + "_ClusMethod_" + str(cluster_method) + "_at_" + str(args.layer) + '_label_' + str(args.label) + "_top_" + str(args.top) + ".pkl"
    return name


def gen_filesaves(args):
    names = []
    for method in args.explainer:
        name = RESULT_DIR + method + '_' + args.mode + '_' + str(args.label) + '_' + str(args.layer) + ".npy"
        names.append(name)
    return names


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
        
        neuron_score = neuron_score.reshape((256,1,1))
        
        if args.layer == None: # if layer is the input, convert to input dimension
            PIL_image = Image.fromarray(neuron_score[0])
            PIL_image = PIL_image.resize((32,32), Image.NEAREST)
            neuron_score = np.array(PIL_image)

#         if args.top != None: # if top is specified, return 0-1 mask with top k neurons
        mask = np.zeros_like(neuron_score)
        mask[largest_indices(neuron_score, args.top)] = 1
        neuron_score = mask
        
        # Normalized
        neuron_score = (neuron_score - np.min(neuron_score))/(np.max(neuron_score) - np.min(neuron_score))
        neuron_score = neuron_score / np.sqrt(np.sum(neuron_score**2))
        score_list.append(neuron_score)
            
    score_dict = dict(zip(args.explainer, score_list))
    return score_dict


def arg_parse():
    parser = argparse.ArgumentParser(description="MNIST class important neurons.")
    parser.add_argument(
            "--mode", dest="mode", help="Train mode: base2 or distill"
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
            "--batchsize", dest="batch_size", type=int, help="Batch size."
        )
    parser.add_argument(
            "--layer", dest="layer", type=int, help="Layer to attribute (12 or 13)."
        )

    parser.set_defaults(
        mode = "base2",
        explainer = ["FDR", "Saliency", "IntegratedGradients", "DeepLift", "GradientSHAP"],
        label = 0,
        top = 10,
        batch_size = 30000,
        layer = 12
    )
    return parser.parse_args()

args = arg_parse()

print("Unsupervised learning on observed activations at ", args.layer)
print("Number of keeping neurons: ", args.top)

batch_size = args.batch_size
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
mode = args.mode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
baseset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=batch_size, shuffle=False, num_workers=2)

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
    # testset.targets = torch.LongTensor(testset.targets)
    # for target in not_animals:
    #     testset.targets[testset.targets == target] = 0
    # for target in animals:
    #     testset.targets[testset.targets == target] = 1


trainset.targets = trainset.targets.clone().detach()
idx = trainset.targets == args.label
trainset.targets = trainset.targets[idx]
trainset.data = trainset.data[idx.numpy().astype(bool)]
print("Number of samples in train set: ", len(trainset))
baseset.targets = torch.LongTensor(baseset.targets)
baseset.targets = baseset.targets[idx]
baseset.data = baseset.data[idx.numpy().astype(bool)]
print("Number of samples in base set: ", len(baseset))

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


score_dict = load_score(args)
probing = Prober(net.module)
probing.compute_dataset_activation(trainset, device)
activation = probing.activation

examined_layer = args.layer

if examined_layer == 12:
    X_var = np.array(probing.activation['features_29'].view(len(trainset), -1).cpu())
elif examined_layer == 13:
    X_var = np.array(probing.activation['features_32'].view(len(trainset), -1).cpu())


cluster_methods = ['KMeans', 'GaussianMixture', 'AgglomerativeClustering']
no_k_components = np.asarray(range(30))+1

base_label = baseset.targets.cpu().detach().numpy()

for cluster_method in cluster_methods:
    print("Running ", cluster_method)
    entropies = {}
    entropies['k'] = no_k_components
    for method in args.explainer:
        entropies[method] = []
    
    for no_k_component in no_k_components:
        for method in args.explainer:
            X_neu = X_var[:, score_dict[method].nonzero()[0]]

            if cluster_method == 'GaussianMixture':
                cmodel = mixture.GaussianMixture(n_components=no_k_component, covariance_type='full')
            elif cluster_method == 'KMeans':
                cmodel = cluster.KMeans(n_clusters=no_k_component)
            elif cluster_method == 'SpectralClustering':
                cmodel = cluster.SpectralClustering(n_clusters=no_k_component)
            elif cluster_method == 'AgglomerativeClustering':
                cmodel = cluster.AgglomerativeClustering(n_clusters=no_k_component)

            cluster_label = cmodel.fit_predict(X_neu)

            label_entropy = []
            for c in range(no_k_component):
                match_label = base_label[cluster_label == c]
                if match_label.size == 0:
                    label_entropy.append(0)
                else:
                    label_entropy.append(drv.entropy(match_label))

            entropy = np.mean(label_entropy)
            entropies[method].append(entropy)
            
    for method in args.explainer:
        entropies[method] = np.asarray(entropies[method])
    entropy_pd = pd.DataFrame.from_dict(entropies)
    filesave = gen_panda_save(args, cluster_method)
    print("Saving file at: ", filesave)
    entropy_pd.to_pickle(filesave)