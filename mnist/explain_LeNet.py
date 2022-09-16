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
from captum.attr import GuidedGradCam
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from skimage.segmentation import slic

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
    
    parser.set_defaults(
        mode = "base",
        explainer = "Saliency",
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

N_EPOCHS = 15
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

if explanation_method == 'Saliency':
    explainer = Saliency(net)
elif explanation_method == 'IntegratedGradients':
    explainer = IntegratedGradients(net)
elif explanation_method == 'DeepLift':
    explainer = DeepLift(net)
elif explanation_method == 'SmoothGrad':
    explainer = IntegratedGradients(net)
elif explanation_method == "GradientSHAP":
    explainer = GradientShap(net)
elif explanation_method == "GuidedGradCam":
    explainer = GuidedGradCam(net, net.feature_extractor[6])
elif explanation_method == "Lime":
    exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
    explainer = Lime(
        net, 
        interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
        similarity_func=exp_eucl_distance
    )
else:
    explainer = None

score = np.zeros(valid_loader.dataset[0][0].shape)
print("Number of samples: ", len(valid_loader)*batch_size)
start_time = time.time()
for inputs, labels in valid_loader:
    outputs = net(inputs.to(DEVICE))
    _, predicted = torch.max(outputs.data, 1)

    for ind in range(len(inputs)):
        img = inputs[ind].unsqueeze(0)
        img.requires_grad = True
        img = img.to(DEVICE)

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
        elif explanation_method == "GuidedGradCam":
            grads = explainer.attribute(img, target=prog_args.label)
            score = score + grads.squeeze().cpu().detach().numpy()
        elif explanation_method == "Lime":
            segments = slic(np.transpose(img.squeeze(0).cpu().detach().numpy(), (1,2,0)), 
                            n_segments=16, compactness=4, sigma=1, start_label=0)
            attrs = explainer.attribute(img,
                                    target=torch.tensor(prog_args.label).unsqueeze(0).to(DEVICE),
                                    feature_mask = torch.from_numpy(segments).unsqueeze(0).to(DEVICE),
                                    n_samples=50,
                                    perturbations_per_eval=10,
                                    show_progress=False
                                    )            
            score = score + attrs.squeeze().cpu().detach().numpy()
        else:
            score = score

all_duration = time.time() - start_time
start_time = time.time()
print("Explaning duration: ", all_duration)
    
score = score/len(valid_loader)/batch_size

def gen_filesave(args):
    method = args.explainer
    test_label = args.label
    if args.mode == 'base':
        name = 'result/'+ method + '_' + str(test_label) + ".npy"
    else:
        name = 'result/'+ args.mode + "_" + method + '_' + str(test_label) + ".npy"
    return name

savename = gen_filesave(prog_args)
np.save(savename, score)

def gen_logfile(args):
    method = args.explainer
    test_label = args.label
    if args.mode == 'base':
        name = 'result/'+ method + '_' + str(test_label) + ".txt"
    else:
        name = 'result/'+ args.mode + "_" + method + '_' + str(test_label) + ".txt"
    return name

logname = gen_logfile(prog_args)
with open(logname, 'w') as f:
    f.write(str(all_duration) + "\n")