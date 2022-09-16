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
    
    parser.set_defaults(
        mode = "base",
        explainer = "FDR",
        label = 0,
        fdr = 0.1,
        runs = 10,
        batch_size = 32,
    )
    return parser.parse_args([])

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


def gen_filesave(args):
    method = args.explainer
    test_label = args.label    
    name = 'result/'+ method + '_' + str(test_label) + ".npy"
    return name

loadname = gen_filesave(prog_args)
data = np.load(loadname)