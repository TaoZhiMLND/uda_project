import torch
import torchvision.models as models
from torch import nn
from collections import OrderedDict

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16_bn(pretrained=True)


models_container = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

def get_pretrained_model(arch, hidden_units, output_size):

    #Load the pretrained model from pytorch
    model = models_container[arch]
    # Freeze training for all layers
    for param in model.features.parameters():
        param.require_grad = False
    # Newly created modules have require_grad=True by default
    num_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_features, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout()),
                          ('fc2', nn.Linear(hidden_units, hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout()),
                          ('fc3', nn.Linear(hidden_units, output_size)),
                          ]))
    model.classifier = classifier

    return model

def get_model_saved(arch, filepath):
    checkpoint = torch.load(filepath)
    #Load the pretrained model from pytorch
    #model = models[checkpoint['arch']]
    model = models['vgg']
    # Freeze training for all layers
    for param in model.features.parameters():
        param.require_grad = False
    # Newly created modules have require_grad=True by default
    num_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(num_features, checkpoint['hidden_units'])),
                        ('relu1', nn.ReLU()),
                        ('drop1', nn.Dropout()),
                        ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['hidden_units'])),
                        ('relu2', nn.ReLU()),
                         ('drop2', nn.Dropout()),
                        ('fc3', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
                        ]))
    model.classifier = classifier
    # Load state_dict
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['class_to_idx']