import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import LeNet



def arg_parse():
    parser = argparse.ArgumentParser(description="MNIST training.")
    parser.add_argument(
            "--mode", dest="mode", help="Train mode: base, mod5 or mod2"
        )
    parser.add_argument(
            "--batchsize", dest="batch_size", type=int, help="Batch size."
        )
    parser.set_defaults(
        mode = "base",
        batch_size = 32,
    )
    return parser.parse_args()

prog_args = arg_parse()

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = prog_args.batch_size
mode = prog_args.mode
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
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

torch.manual_seed(RANDOM_SEED)

if mode == "mod2" or mode == "mod5" or mode == "base":
    print("Train base model.")
    model = LeNet.LeNet5(N_CLASSES).to(DEVICE)
else:
    print("Train distill model with 2 labels.")
    model_base = LeNet.LeNet5(10).to(DEVICE)
    PATH_10 = 'pretrained/mnist_base'
    model_base.load_state_dict(torch.load(PATH_10))
    model_base.eval()
    model = LeNet.LeNet5(N_CLASSES).to(DEVICE)
    main_layers = []
    for item in model._modules.items():
        main_layers.append(item[0])
    for layer in main_layers:
        for i in range(len(getattr(model, layer))):
            if layer == 'classifier':
                if i == len(getattr(model, layer)) -1:
                    break
            if hasattr(getattr(model, layer)[i], 'weight'):
                with torch.no_grad():
                    getattr(model, layer)[i].weight.copy_(getattr(model_base, layer)[i].weight)
            if hasattr(getattr(model, layer)[i], 'bias'):
                with torch.no_grad():
                    getattr(model, layer)[i].bias.copy_(getattr(model_base, layer)[i].bias)
    
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("Training: ", mode)
model, optimizer, losses = LeNet.training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

SAVE_PATH = 'pretrained/mnist'
if mode == "mod2":
    SAVE_NAME = '_mod2'
elif mode == "distillmod2":
    SAVE_NAME = '_distillmod2'
elif mode == "mod5":
    SAVE_NAME = '_mod5'
elif mode == "distillmod5":
    SAVE_NAME = '_distillmod5'
else:
    SAVE_NAME = '_base'
torch.save(model.state_dict(), SAVE_PATH + SAVE_NAME)