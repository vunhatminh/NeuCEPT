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
            for layer_index in range(len(getattr(model, layer))):
                getattr(self.model, layer)[layer_index].register_forward_hook(self.get_activation(self.model, layer + "_" + str(layer_index)))

    def get_activation(self, model, name):
        def hook(model, input, output_):
            self.activation[name] = output_.detach()
        return hook 
        
    def forward(self, input_):
        with torch.no_grad():
            self.activation['out'] = self.model(input_).clone()
            self.activation['in'] = input_.clone()
        return self.activation['out']
        
    
    def compute_dataset_activation(self, dataset, device = None):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                                     shuffle=False, num_workers=2)
        dataiter = iter(dataloader)
        images, _ = dataiter.next()
        if device == 'cuda':
            images = images.to(device)
        self.activation['out'] = self.forward(images).clone()
        self.activation['in'] = images.clone()
        