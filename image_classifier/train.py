import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import classifier

import os
import copy
import time
import argparse


TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
data_dir = ''
in_arg = {}
# Main program function defined below
def main():
    # Creates & retrieves Command Line Arugments
    global in_arg
    in_arg = get_input_args()
    global data_dir
    data_dir = in_arg.dir
    print(in_arg)
    check_input_args(in_arg)

    data_holder = load_data(in_arg.dir)

    model = get_trained_model(data_holder, in_arg.arch, in_arg.hidden_units, in_arg.gpu, in_arg.epochs, in_arg.learning_rate)
    
    save_checkpoint(in_arg.arch, model, in_arg.hidden_units, len(data_holder['class_names']),  data_holder['image_datasets'], in_arg.save_dir)

def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='path to folder of data')
    parser.add_argument('--arch', type=str, default = 'vgg')
    parser.add_argument('--save_dir', type=str, default = 'checkpoint.pth')
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--hidden_units', type=int,  default = 4096) 
    parser.add_argument('--epochs', type=int, default = 20)
    parser.add_argument('--gpu', action='store_true', default = False)
    
    # returns parsed argument collection
    return parser.parse_args()

def check_input_args(in_arg):
    # Check input args
    gpu_available = torch.cuda.is_available()
    if ((in_arg.gpu == True) and (gpu_available == False)):
        raise Exception("GPU is not available!")
   
def load_data(data_dir):
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        VALID: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, VALID, TEST]
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=64, shuffle=True
        )
        for x in [TRAIN, VALID, TEST]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VALID, TEST]}

    class_names = image_datasets[TRAIN].classes

    return {'image_datasets' : image_datasets, 'dataloaders' : dataloaders, 'dataset_sizes':dataset_sizes, 'class_names' : class_names}



def get_trained_model(data_holder, arch, hidden_units, use_gpu, epochs, learning_rate):

    print("arch:{}".format(arch))

    model = classifier.get_pretrained_model(arch, hidden_units, len(data_holder['class_names']))

    model = train_model_epochs(model, data_holder['dataloaders'], data_holder['dataset_sizes'], epochs, learning_rate, use_gpu)

    return model

def train_model_epochs(model, dataloaders, dataset_sizes, epochs, learning_rate, use_gpu):
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(dataloaders[TRAIN])
    valid_batches = len(dataloaders[VALID])
        
    if use_gpu:
        model.to('cuda')
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    for e in range(epochs):
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        model.train(True)
        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 20 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
            
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
            # clear gradients
            optimizer.zero_grad()

            #forward pass
            outputs = model(inputs)

            # calculate loss
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            # back propagation
            loss.backward()
            optimizer.step()

            # calculate training loss
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data).item()
            
            del inputs, labels, outputs, preds, loss
            torch.cuda.empty_cache()
            
        avg_loss = loss_train / dataset_sizes[TRAIN]
        avg_acc = acc_train / dataset_sizes[TRAIN]
        
        model.train(False)
        model.eval()
        
        for i, data in enumerate(dataloaders[VALID]):
            if i % 20 == 0:
                print("\rValidation batch {}/{}".format(i, valid_batches), end='', flush=True)

            inputs, labels = data

            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
        
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data).item()
            
            del inputs, labels, outputs, preds, loss
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / dataset_sizes[VALID]
        avg_acc_val = acc_val / dataset_sizes[VALID]
        
        print()
        print("Epoch {} result: ".format(e))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def save_checkpoint(arch, model, hidden_units, output_size, image_datasets, save_dir):
    checkpoint = {
        'arch': arch,
        'hidden_units' : hidden_units,
        'output_size' : output_size,
        'class_to_idx': image_datasets['train'].class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir)


# Call to main function to run the program
if __name__ == "__main__":
    main()