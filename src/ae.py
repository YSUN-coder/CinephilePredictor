#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:26:47 2020

@author: starry
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset, open dataset source: https://grouplens.org/datasets/movielens/
import os

ds_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
movies = pd.read_csv(ds_dir+'/datasets/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(ds_dir+'/datasets/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(ds_dir+'/datasets/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv(ds_dir+'/ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv(ds_dir+'/ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the num of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users] # all movies for one user
        id_ratings = data[:, 2][data[:, 0] == id_users] # all ratings for one user
        ratings = np.zeros(nb_movies) # len(id_movies)==len(id_ratings) len(rating) == len(nb_movies)
        ratings[id_movies - 1] = id_ratings  #np.zeros(), then mapping two list into ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Convert the numpy data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
        
# Create the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # start to build AutoEncoder layers with full connection layers
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20,nb_movies)
        self.activation = nn.Sigmoid() # ?
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : 2D vector
            encoding it twice and decoding it twice 
            to get the final reconstructed output vector.

        Returns
        -------
        vector of predicted ratings.

        '''
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
sae = SAE()
# loss function to measure the mean squard error
criterion = nn.MSELoss() # ?
# GD root-mean-squared error (float)
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5) # Adam/SGD
 # Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # add a new D corresponding to batches
        # After each observation, w will be changed
        target = input.clone()
        if torch.sum(target.data > 0) > 0: # torch help collect data!=0
            output = sae(input) # nn.Module will call forward()
            target.required_grad = False # make sure the gradient not with respect to the target, which save lots of computations and optimizes the code
            output[target==0] = 0 # target==0 will not count on error to save computation
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # 1e-10 for avoiding 0 and small num only bring less bias
            loss.backward()  # weight update direction +/-
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()  # weight update intensity
    print('epoch: ' + str(epoch) + " loss:"+ str(train_loss/s))
    
# Test the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # add a new D corresponding to batches
    target = Variable(test_set[id_user]).unsqueeze(0)
    
    if torch.sum(target.data > 0) > 0: # torch help collect data!=0
        output = sae(input) # nn.Module will call forward()
        target.required_grad = False # make sure the gradient not with respect to the target, which save lots of computations and optimizes the code
        output[target==0] = 0 # target==0 will not count on error to save computation
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # 1e-10 for avoiding 0 and small num only bring less bias
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print("Test loss:"+ str(test_loss/s))   
