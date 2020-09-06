#MODULES USED
# Imports here
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import torch
import PIL
import numpy as np
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models


# ARCH STRUCTURE SUPPORTED
arch = {"vgg16":25088,
        "densenet121":1024
        }

def transform_load_data(root_dir):
    data_dir = root_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms_test_valid = transforms.Compose([transforms.Resize(225),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)
    image_datasets_test = datasets.ImageFolder(data_dir + '/test', transform=data_transforms_test_valid)
    image_datasets_valid = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms_test_valid)
    image_datesets_dict = {"train":image_datasets,"test":image_datasets_test,"valid":image_datasets_valid}
    #Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = torch.utils.data.DataLoader(image_datasets,batch_size = 64,shuffle=True)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size = 64)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid,batch_size = 64)
    data_loader_dict = {"train_dl":dataloaders_train,"test_dl":dataloaders_test,"valid_dl":dataloaders_valid}
    return image_datesets_dict,data_loader_dict



def model_constructor(arch_chosen ='vgg16',dropout = 0.2,hidden_layers = 1500,learning_rate = 0.001,device_chosen = "gpu" ):
    device = torch.device("cuda" if torch.cuda.is_available() and device_chosen=="gpu" else"cpu")
    if arch_chosen == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch_chosen == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Only vgg16 or densenet121 Are Supported")
        
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(arch[arch_chosen],hidden_layers),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_layers,102),
                                 nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),learning_rate)
    model.to(device);
#     if torch.cuda.is_available() and device_chosen=="gpu":
#         model.cuda()
    return model,criterion,optimizer


def train_model_process(chosen_model,criterion, optimizer,epochs = 1, print_every=5, device='gpu'):
    img_datasets,data_loaders = transform_load_data("./flowers/")
#     epochs = 1
    steps = 0
    running_loss = 0
#     print_every = 5
    for epoch in range(epochs):
        for inputs, labels in tqdm(data_loaders['train_dl']):
            steps += 1
            # Move input and label tensors to the default device
#             inputs, labels = inputs.to(device), labels.to(device)
            if torch.cuda.is_available() and device =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')


            optimizer.zero_grad()

            logps = chosen_model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                chosen_model.eval()
                with torch.no_grad():
                    for inputs, labels in data_loaders['valid_dl']:
#                         inputs, labels = inputs.to(device), labels.to(device)
                        if torch.cuda.is_available() and device =='gpu':
                            inputs, labels = inputs.to('cuda'), labels.to('cuda')

                        logps = chosen_model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Dataset Loss: {running_loss/print_every:.3f}.. "
                      f"Valid Dataset Loss: {valid_loss/len(data_loaders['valid_dl']):.3f}.. "
                      f"Valid accuracy: {accuracy/len(data_loaders['valid_dl']):.3f}")
                running_loss = 0
                chosen_model.train()
                
                
def save_checkpoint(chosen_model = 0 ,PATH = "checkpoint.pth",chosen_arch ='vgg16', hidden_layers = 1500,dropout=0.2,learning_rate=0.001,epochs=1):
        
    img_datasets,data_loaders = transform_load_data("./flowers/")
    chosen_model.class_to_idx = img_datasets['train'].class_to_idx
    chosen_model.cpu
    # def save_checkpoint(model,input_size,output_size,epoch,class_to_idx,learning_rate,PATH):
    checkpoint ={
                'epoch': epochs,
                'model_state_dict': chosen_model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
                "classifier": chosen_model.classifier,
                'learning_rate': learning_rate,
                "hidden_layer1":hidden_layers,
                "arch": chosen_arch,
                "dropout": dropout,
                "class_to_idx" : chosen_model.class_to_idx
        }
    torch.save(checkpoint, PATH)

def load_checkpoint(filepath = './checkpoint.pth'):
    checkpoint = torch.load(filepath)
    epoch = checkpoint['epoch']
    classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    arch = checkpoint['arch']
    dropout = checkpoint['dropout']
    hidden_layers = checkpoint['hidden_layer1']
    class_to_idx =checkpoint['class_to_idx']
    
    model,criterion,optimizer = model_constructor(arch,dropout,hidden_layers,learning_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, class_to_idx

def process_image(image_path = './flowers/valid/29/image_04104.jpg'):
    
    img = Image.open(image_path)
    img_data_transforn = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    image = img_data_transforn(img)
    return image

def predict(image_path, model=0, topk=5,device='gpu'):
   
    if torch.cuda.is_available() and device =='gpu':
        model.to('cuda')
    img = process_image(image_path)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    gpu_input = img.cuda()
    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(gpu_input)
    else:
        with torch.no_grad():
            output=model.forward(img)
    
#     model.eval()
#     inputs = Variable(img).to(device)
#     logits = model.forward(inputs)
    
    ps = F.softmax(output,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)









