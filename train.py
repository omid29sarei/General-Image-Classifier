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

#Defining the Argure Parser Object
argue_parser = argparse.ArgumentParser(description='Train.py')

#Adding Argument to Argue Parser Object
argue_parser.add_argument('data_dir', action="store", default="./flowers/")
argue_parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
argue_parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
argue_parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
argue_parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
argue_parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
argue_parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
argue_parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=1500)

# Extracting information from the Argue Parser
parsed_argue = argue_parser.parse_args()
root_data = parsed_argue.data_dir
checkpoint_path = parsed_argue.save_dir
learning_rate = parsed_argue.learning_rate
chosen_arch = parsed_argue.arch
dropout = parsed_argue.dropout
hidden_layers = parsed_argue.hidden_units
device_chosen = parsed_argue.gpu
epochs = parsed_argue.epochs


def main():

    img_datasets,data_loaders = utilities.transform_load_data(root_data)
    model,criterion,optimizer = utilities.model_constructor(chosen_arch,dropout,hidden_layers,learning_rate,device_chosen)
    utilities.train_model_process(model,criterion,optimizer)
    print("Training Done...")
    utilities.save_checkpoint(model,checkpoint_path,chosen_arch,hidden_layers,dropout,learning_rate,device_chosen)
    print("Checkpoint Saved...")

if __name__ == "__main__":
    main()
