import torch

class Prober:
    def __init__(self, model, layers = None):
        self.model = model
        self.model.eval()
        self.activation = {}
        self.activation['out'] = None
        self.activation['in'] = None
        if layers == None:
            self.layers = []
            for item in model._modules.items():
                self.layers.append(item[0])
        else:
            self.layers = layers
            
        for layer in self.layers:
            if layer == 'features':
                for layer_index in range(len(getattr(model, layer))):
                    getattr(self.model, layer)[layer_index].register_forward_hook(self.get_activation(self.model, layer + "_" + str(layer_index)))
            elif layer == 'classifier':
                self.model.classifier.register_forward_hook(self.get_activation(self.model, layer))
                

    def get_activation(self, model, name):
        def hook(model, input, output_):
            self.activation[name] = output_.detach()
        return hook 
        
    def forward(self, input_):
        with torch.no_grad():
            self.activation['out'] = self.model(input_).clone()
            self.activation['in'] = input_.clone()
        return self.activation['out']
        
    
    def compute_dataset_activation(self, dataset, device=None, batch_size=None):
        if batch_size == None:
            batch_size = len(dataset)
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  # batch norm can only handle up to 30000
                                                     shuffle=False, num_workers=4)
        dataiter = iter(dataloader)
        images, label = dataiter.next()
        if device == 'cuda':
            images = images.to(device)
        self.activation['out'] = self.forward(images).clone()
        self.activation['in'] = images.clone()
        
        return label
        
    def print_activation_shape(self, dataset, device=None):
        self.compute_dataset_activation(dataset, device)
        for layer in self.activation.keys():
            print(layer, self.activation[layer].shape)
        
        
        