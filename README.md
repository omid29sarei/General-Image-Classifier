# General-Image-Classifier
This is a console application of  image classifier where users can choose the model between two supported architecture

# Prerequisites
for the purpose of this project libraries such as numpy,pandas,matplotlib and pil where used so in order to make use of them
you have to install them either throuh pip command or directly through conda

# HOW TO USE
1- navigate to cloned project.
2- train.py is used for training the model.
3- predict.py is used to predict the image and the top 5 category names.
4- utilities.py is used in order to structure the code base. all the functions can be found in this file.
5- first run train.py with the arguments you want and then run the predict with chosen arguments and it should output the result in the console.
6- ENJOY


# Command Line Application Usage

Basic usage: python train.py data_directory ex: ./flower/

# Supported Arguments
--save_dir will let you to specify a directory
--arch will let you specify which architecture want to use ex: vgg16
--hidden_layers will let you specify the amount of hidden layers you want to have. ex: 1024
--learning_rate will let you specify the learning rate. ex: 0.001
--epochs will let you specify the amount of epochs



