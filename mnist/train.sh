#!/bin/bash
## TRAINING

python train_LeNet.py --mode base
python train_LeNet.py --mode mod2
python train_LeNet.py --mode mod5
python train_LeNet.py --mode distillmod2
python train_LeNet.py --mode distillmod5
