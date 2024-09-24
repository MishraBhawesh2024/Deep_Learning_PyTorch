#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.init as init


# In[15]:


df1 = pd.read_csv('water_potability.csv')


# In[16]:


df1.shape


# In[17]:


df1.dropna(inplace = True)
df1.shape


# In[18]:


from torch.utils.data import Dataset


# In[19]:


# Implementing a Waterset class that inherits from the Dataset class

class Waterset(Dataset):
    
    def __init__(self, csv_path):
        super().__init__()             # maintaining inheritance from Dataset
        df = pd.read_csv(csv_path)
        df.dropna(inplace = True)     # already inspected and dropping the rows with null values
        self.data = df.to_numpy()    # data is a numpy array to feed into ML/DL models
    
    
    def __len__(self):
        
        return self.data.shape[0]
    
    def __getitem__(self, idx):  # returns features and labels for a given index idx
        
        features = self.data[idx, : -1]
        features = features.astype('float32')      # cast features as float32 because weights are float32 in Torch
        
        label = self.data[idx, -1]
        label = label.astype('float32')
        
        label = label.reshape(-1, 1)               # turn label into an array
        
        return features, label
    


# In[20]:


data = Waterset('water_potability.csv')


# In[70]:


from torch.utils.data import DataLoader

# since we are using batchnormalization later, we use drop_last = True, 
# in case the number of rows in data is not divisible by batch_size - because batchnormalization with default parameters 
# keeps track of running_mean and std_dev

dataloader = DataLoader(data, batch_size = 3, shuffle = True, drop_last = True)


# In[58]:


print(feature_sample.dtype, label_sample.dtype)


# In[59]:


len(data)


# In[68]:


import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()               # inheritance from parent class
        self.fc1 = nn.Linear(9, 16)               # input layer
        init.kaiming_uniform_(self.fc1.weight)    # using kaiming initialization of weights
        self.bn1 = nn.BatchNorm1d(16)             # batchnormalization
        
        # above steps for the second layer (hidden layer)
        # bactchnormalization facilitates faster learning and prevents the problem of unstable gradients too
        
        self.fc2 = nn.Linear(16, 8)               
        init.kaiming_uniform_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(8)
        
        
        self.fc3 = nn.Linear(8, 1)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity = 'sigmoid')
        
        # in the last layer, we add nonlinearity for binary classification task at hand
    
    def forward(self, x):
        
        # using elu function so that we do not have inactive dying neurons
        
        x = self.fc1(x)
        x = F.elu(x)
        x = self.bn1(x)
        
        x = self.fc2(x)
        x = F.elu(x)
        x = self.bn2(x)
        
        x = self.fc3(x)
        x = F.sigmoid(x)
    
        
        return x


# In[69]:


import torch.optim as optim
net = Net()


criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001)
accuracies = [0]*10

for epoch in range(10):
    
    total_correct, total_samples = 0, 0
    
    for features, labels in dataloader:
        
        optimizer.zero_grad()                                                 # reset the gradient for each batch
        outputs = net(features)                                       # forward propagation
        loss = criterion(outputs, labels.view(-1, 1))                         # calculate loss
        loss.backward()                                                       # calculate gradients
        optimizer.step()                                                      # update parameters
        
        predicted = (outputs >= 0.5).float()                                  # convert to binary 
        
        # since labels is a 3d tensor, convert it into an array using labels.view(-1, 1) 
        # and check for how many of them are equal to labels. Since the answer is still a tensor, convert it into a float using 
        # .item()
        
        total_correct += (predicted == labels.view(-1, 1)).sum().item()       
        total_samples += 3
        
    # Calculate the accuracy for this epoch
    accuracy = 100 * total_correct / total_samples
    accuracies.append(accuracy)
    print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')
    
        
                                                           


# In[ ]:




