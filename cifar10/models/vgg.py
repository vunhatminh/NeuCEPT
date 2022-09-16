'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 96, 'M', 128, 128, 'M', 256, 256, 'M', 256, 'M', 256, 256],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, n_classes, vgg_name='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(256, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_A(nn.Module):
    def __init__(self, n_classes, layer=9, vgg_name='VGG11'):
        super(VGG_A, self).__init__()
        self.end_layer = layer
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if i == self.end_layer + 1:
                break
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def load_from_VGG(self, net):
        net.eval()
        self.eval()
        features = self.features
        features_net = net.features

        for i in range(len(features)):
            if hasattr(features[i], 'weight'):
                with torch.no_grad():
                    features[i].weight.copy_(features_net[i].weight)
            if hasattr(features[i], 'bias'):
                with torch.no_grad():
                    features[i].bias.copy_(features_net[i].bias)
            if hasattr(features[i], 'running_mean'):
                with torch.no_grad():
                    features[i].running_mean.copy_(features_net[i].running_mean)
            if hasattr(features[i], 'running_var'):
                with torch.no_grad():
                    features[i].running_var.copy_(features_net[i].running_var)


class VGG_B(nn.Module):
    def __init__(self, n_classes, layer=9, vgg_name='VGG11'):
        super(VGG_B, self).__init__()
        self.end_layer = layer
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                if i > self.end_layer:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i > self.end_layer:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def load_from_VGG(self, net):
        net.eval()
        self.eval()
        features = self.features
        features_net = net.features

        for i in range(len(features)):
            if hasattr(features[-i-1], 'weight'):
                with torch.no_grad():
                    features[-i-1].weight.copy_(features_net[-i-1].weight)
            if hasattr(features[-i-1], 'bias'):
                with torch.no_grad():
                    features[-i-1].bias.copy_(features_net[-i-1].bias)
            if hasattr(features[-i-1], 'running_mean'):
                with torch.no_grad():
                    features[-i-1].running_mean.copy_(features_net[-i-1].running_mean)
            if hasattr(features[-i-1], 'running_var'):
                with torch.no_grad():
                    features[-i-1].running_var.copy_(features_net[-i-1].running_var)

        with torch.no_grad():
            self.classifier.weight.copy_(net.classifier.weight)
            self.classifier.bias.copy_(net.classifier.bias)

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
