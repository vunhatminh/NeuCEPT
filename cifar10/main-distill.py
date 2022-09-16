'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.vgg import *
from utils import progress_bar

PRETRAINED_DIR = './pretrained/'
assert os.path.isdir(PRETRAINED_DIR), 'Error: no pretrained directory found!'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument(
        "--mode", dest="mode", help="Train mode: base10, base2 or distill, "
    )
parser.add_argument(
        "--batchsize", dest="batch_size", type=int, help="Batch size."
    )
parser.set_defaults(
    mode = "base10",
    batch_size = 4,
)
args = parser.parse_args()

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
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

animals = [2,3,4,5,6,7]
not_animals = [0,1,8,9]

mode = args.mode

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

# Model
print('==> Building model..')

if mode == "base10":
    net = VGG(10, 'VGG11')
elif mode == "base2" or mode == 'distill':
    net = VGG(2, 'VGG11')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if mode == 'base10':
        checkpoint = torch.load(PRETRAINED_DIR + 'cifar-base10.pth')
    elif mode == 'base2':
        checkpoint = torch.load(PRETRAINED_DIR + 'cifar-base2.pth')
    elif mode == 'distill':
        checkpoint = torch.load(PRETRAINED_DIR + 'cifar-distill.pth')

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    
for i in range(21):
    for param in net.module.features[i].parameters():
        param.requires_grad = False
# for param in net.module.features.parameters():
#     param.requires_grad = False
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if mode == 'distill':
    print('==> Training distill model with 2 labels')
    # Load base10 model
    net10 = VGG(10, 'VGG11').to(device)
    net10 = torch.nn.DataParallel(net10)
    net10.load_state_dict(torch.load(PRETRAINED_DIR + 'cifar-base10.pth')['net'])
    net10.eval()

    features_distill = net.module.features
    features10 = net10.module.features

    for i in range(len(features_distill)):
        if hasattr(features_distill[i], 'weight'):
            with torch.no_grad():
                features_distill[i].weight.copy_(features10[i].weight)
        if hasattr(features_distill[i], 'bias'):
            with torch.no_grad():
                features_distill[i].bias.copy_(features10[i].bias)
        if hasattr(features_distill[i], 'running_mean'):
            with torch.no_grad():
                features_distill[i].running_mean.copy_(features10[i].running_mean)
        if hasattr(features_distill[i], 'running_var'):
            with torch.no_grad():
                features_distill[i].running_var.copy_(features10[i].running_var)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if mode == "base10":
            path = PRETRAINED_DIR + 'cifar-base10.pth'
        elif mode == "base2":
            path = PRETRAINED_DIR + 'cifar-base2.pth'
        elif mode == 'distill':
            path = PRETRAINED_DIR + 'cifar-distill-freeze13.pth'
        torch.save(state, path)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
