#MODULES USED
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import torch
import PIL
import numpy as np
import torchvision
import argparse
from PIL import Image
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

#LOCAL IMPORT
import utilities


argue_parser = argparse.ArgumentParser(description='Predict.py')

argue_parser.add_argument('input', default='./flowers/valid/29/image_04104.jpg', nargs='?', action="store", type = str)
argue_parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
argue_parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
argue_parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
argue_parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
argue_parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

parsed_argue = argue_parser.parse_args()
path_image = parsed_argue.input
number_of_outputs = parsed_argue.top_k
device = parsed_argue.gpu
file_path = parsed_argue.checkpoint
data_directory = parsed_argue.data_dir
category_names = parsed_argue.category_names

def main():
    model , class_to_idx = utilities.load_checkpoint(file_path)
    map_location='cpu'
    strict=False
    img_datasets,data_loaders = utilities.transform_load_data(data_directory)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    probs, classes = utilities.predict(path_image, model,number_of_outputs,device)
    print("THERE ARE THE PROBABILITIES: ",probs)
    print("THESE ARE THE CLASSES: ",classes)
    class_names = img_datasets['train'].classes
    flower_names = [cat_to_name[class_names[e]] for e in classes]
    print("THESE ARE THE TOP 5 CATEGORY FLOWER NAMES: ",flower_names)
    
    
if __name__ == "__main__":
    main()


