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
from pyitlib import discrete_random_variable as drv
from sklearn import cluster
from sklearn import mixture

def gen_filesaves(args):
    names = []
    if args.mode == 'base':
        for method in args.explainer:
            if args.layer == None:
                name = 'result/'+ method + '_' + str(args.label) + ".npy"
            else:
                name = 'result/'+ method + '_' + str(args.label) + '_' + args.layer + ".npy"
            names.append(name)
    else:
        for method in args.explainer:
            if args.layer == None:
                name = 'result/'+ args.mode + "_" + method + '_' + str(args.label) + ".npy"
            else:
                name = 'result/'+ args.mode + "_" + method + '_' + str(args.label) + '_' + args.layer + ".npy"
            names.append(name)
    return names

def gen_panda_save(args, cluster_method):
    if args.layer == None:
        name = 'result/' + args.mode + "_ClusMethod_" + str(cluster_method) + "_top_" + str(args.top) + ".pkl"
    else:
        name = 'result/' + args.mode + "_ClusMethod_" + str(cluster_method) + "_at_" +  args.layer + "_top_" + str(args.top) + ".pkl"
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
            "--batchsize", dest="batch_size", type=int, help="Batch size."
        )
    parser.add_argument(
            "--layer", dest="layer", help="Layer to attribute ('linear0' or 'conv3')."
        )

    parser.set_defaults(
        mode = "distillmod2",
        explainer = ["FDR", "Saliency", "IntegratedGradients", "DeepLift", "GradientSHAP"],
        label = 0,
        top = 10,
        batch_size = 32,
        layer = 'linear0'
    )
    return parser.parse_args()

prog_args = arg_parse()

print("Unsupervised learning on observed activations at ", prog_args.layer)
print("Number of keeping neurons: ", prog_args.top)

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42

batch_size = prog_args.batch_size
mode = prog_args.mode

IMG_SIZE = 32
N_CLASSES = 10

# define transforms
tf = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=tf,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=tf)

base_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=tf,
                               download=True)

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
    
train_dataset.targets = train_dataset.targets.clone().detach()
idx = train_dataset.targets == prog_args.label
train_dataset.targets = train_dataset.targets[idx]
train_dataset.data = train_dataset.data[idx.numpy().astype(bool)]
print("Number of samples in train set: ", len(train_dataset))
base_dataset.targets = base_dataset.targets[idx]
base_dataset.data = base_dataset.data[idx.numpy().astype(bool)]
print("Number of samples in base set: ", len(base_dataset))

valid_dataset.targets = valid_dataset.targets.clone().detach()
idx = valid_dataset.targets == prog_args.label
valid_dataset.targets= valid_dataset.targets[idx]
valid_dataset.data = valid_dataset.data[idx.numpy().astype(bool)]
print("Number of samples in test set: ", len(valid_dataset))
    
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

score_dict = load_score(prog_args)

probing = Prober(net)
probing.compute_dataset_activation(train_dataset, device = DEVICE)
activation = probing.activation

examined_layer = prog_args.layer

if examined_layer == 'linear0':
    X_var = np.array(probing.activation['classifier_1'].view(len(train_dataset),-1).cpu())
elif examined_layer == 'conv3':
    X_var = np.array(probing.activation['feature_extractor_7'].view(len(train_dataset),-1).cpu())
else:
    pre_process = transforms.Compose([transforms.Resize((8,8))])
    X_var = np.array(pre_process(probing.activation['in']).view(len(train_dataset),-1).cpu())
    
cluster_methods = ['KMeans', 'GaussianMixture', 'AgglomerativeClustering']

if mode == "mod2" or mode == "distillmod2":
    no_k_components = np.asarray(range(12))+1
else:
    no_k_components = np.asarray(range(12))+1

base_label = base_dataset.targets.cpu().detach().numpy()

for cluster_method in cluster_methods:
    print("Running ", cluster_method)
    entropies = {}
    entropies['k'] = no_k_components
    for method in prog_args.explainer:
        entropies[method] = []
    
    for no_k_component in no_k_components:
        for method in prog_args.explainer:
            X_neu = X_var[:,score_dict[method].nonzero()[0]]

            if cluster_method == 'GaussianMixture':
                cmodel = mixture.GaussianMixture(n_components=no_k_component, covariance_type='full')
            elif cluster_method == 'KMeans':
                cmodel = cluster.KMeans(n_clusters=no_k_component)
            elif cluster_method == 'SpectralClustering':
                cmodel = cluster.SpectralClustering(n_clusters=no_k_component)
            elif cluster_method == 'AgglomerativeClustering':
                cmodel = cluster.AgglomerativeClustering(n_clusters=no_k_component)

            cluster_label = cmodel.fit_predict(X_neu)
            entropy = np.mean(np.asarray([drv.entropy(np.array(base_label[cluster_label==c])) for c in range(no_k_component)]))
            entropies[method].append(entropy)
            
    for method in prog_args.explainer:
        entropies[method] = np.asarray(entropies[method])
    entropy_pd = pd.DataFrame.from_dict(entropies)
    filesave = gen_panda_save(prog_args, cluster_method)
    print("Saving file at: ", filesave)
    entropy_pd.to_pickle(filesave)