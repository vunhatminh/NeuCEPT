import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime 

from torchvision import datasets, transforms

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            y_hat = model(X)
            y_prob = F.softmax(y_hat, dim=1)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n
    
def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
    
    return model, optimizer, (train_losses, valid_losses)

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
#         probs = F.softmax(logits, dim=1)
#         return logits, probs
        return logits

class LeNet5_A(nn.Module):
    def __init__(self, layer):
        super(LeNet5_A, self).__init__()
        
        self.end_layer = layer
        
        if layer == 'conv2':
            self.feature_extractor = nn.Sequential(            
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh()
            )
        elif layer == 'conv3':
            self.feature_extractor = nn.Sequential(            
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                nn.Tanh()
            )
        elif layer == 'linear0':
            self.feature_extractor = nn.Sequential(            
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                nn.Tanh()
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_features=120, out_features=84),
                nn.Tanh()
            )
        else:
            print("Wrong configuration.")
            
    def load_from_LeNet(self, lenet):
        main_layers = []
        for item in self._modules.items():
            main_layers.append(item[0])
        for layer in main_layers:
            for i in range(len(getattr(self, layer))):
                if hasattr(getattr(self, layer)[i], 'weight'):
                    with torch.no_grad():
                        getattr(self, layer)[i].weight.copy_(getattr(lenet, layer)[i].weight)
                if hasattr(getattr(self, layer)[i], 'bias'):
                    with torch.no_grad():
                        getattr(self, layer)[i].bias.copy_(getattr(lenet, layer)[i].bias)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        if self.end_layer == 'linear0':
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif self.end_layer == 'conv3':
            x = torch.flatten(x, 1)
        return x
    
class LeNet5_B(nn.Module):
    def __init__(self, layer, n_classes):
        super(LeNet5_B, self).__init__()
        
        self.end_layer = layer
        
        if layer == 'conv2':
            self.feature_extractor = nn.Sequential(            
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                nn.Tanh()
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_features=120, out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=n_classes),
            )
        elif layer == 'conv3':
            self.classifier = nn.Sequential(
                nn.Linear(in_features=120, out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=n_classes),
            )
        elif layer == 'linear0':
            self.classifier = nn.Sequential(
                nn.Linear(in_features=84, out_features=n_classes),
            )
        else:
            print("Wrong configuration.")
            
            
            
    def load_from_LeNet(self, lenet):
        main_layers = []
        for item in self._modules.items():
            main_layers.append(item[0])
        for layer in main_layers:
            for i in range(len(getattr(self, layer))):
                if hasattr(getattr(self, layer)[-i-1], 'weight'):
                    with torch.no_grad():
                        getattr(self, layer)[-i-1].weight.copy_(getattr(lenet, layer)[-i-1].weight)
                if hasattr(getattr(self, layer)[-i-1], 'bias'):
                    with torch.no_grad():
                        getattr(self, layer)[-i-1].bias.copy_(getattr(lenet, layer)[-i-1].bias)
        
    def forward(self, x):
        if self.end_layer == 'conv2':
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
        else:
            logits = self.classifier(x)
#         probs = F.softmax(logits, dim=1)
#         return logits, probs
        return logits