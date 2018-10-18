from PIL import Image

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn

import os
import copy
import argparse
import json

import classifier
import image

use_gpu = torch.cuda.is_available()

# Main program function defined below
def main():
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    print(in_arg)
    check_input_args(in_arg)

    model, class_to_idx = classifier.get_model_saved(in_arg.arch, in_arg.checkpoint)
    cat_to_name = get_cat_to_name(in_arg.category_names)

    idx_to_class = {}
    for key, value in class_to_idx.items(): 
        idx_to_class[str(value)] = key

    probs, classes = predict(in_arg.input, model, in_arg.gpu, idx_to_class, in_arg.top_k)
    class_labels = [cat_to_name[x] for x in classes]
    print(probs)
    print(class_labels)

def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', type=int, default = 5) 
    parser.add_argument('--category_names', type=str, default = 'cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default = False)
    
    # returns parsed argument collection
    return parser.parse_args()

def check_input_args(in_arg):
    # Check input args
    if ((in_arg.gpu == True) and (use_gpu == False)):
        raise Exception("GPU is not available!")

def get_cat_to_name(name_json_file):
    with open(name_json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def predict(image_path, model, use_gpu, idx_to_class, topk):
    if use_gpu:
        model.to('cuda')
    model.eval()
    softmax = nn.Softmax(dim=1)
    
    inputs = np.array([image.process_image(image_path)])
    inputs = torch.from_numpy(inputs).float()
    if use_gpu:
        inputs.to('cuda')
    
    outputs = model(inputs)
    outputs = softmax(outputs)
    
    topk = outputs.topk(topk)
    probs = topk[0].to('cpu').detach().numpy()[0]
    indices = topk[1].to('cpu').detach().numpy()[0]
    del inputs, outputs
    torch.cuda.empty_cache()
    
    classes = [idx_to_class[str(x)] for x in indices]
    
    return probs, classes

# Call to main function to run the program
if __name__ == "__main__":
    main()