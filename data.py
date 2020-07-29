import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import random 
from sklearn import metrics
from statistics import mean
from statistics import mode  
from torch.utils import data           
from torch import nn
from torch.nn import functional as F
import math
from torch import optim
import sys
import pickle
import cv2
from sklearn.feature_extraction import image

class numpyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()
    
PATH = "L:/Projects/Apples_vs_Tomatoes/Dataset/"
        
class DATASET(Dataset):
  
  def __init__(self):
      
    apples_path = PATH + "apples.txt"
    tomatoes_path = PATH + "tomatoes.txt"

    # Initialize X (data_names), y (apple=0, tomato=1)
    X = []
    y = []

    # read apples names
    with open(apples_path, 'r') as f:
      apples_names = f.readlines()

    self.n_apples = len(apples_names)

    # append real_data to X and y
    X.extend(apples_names)
    y.extend([0]*self.n_apples)
    
    # read tomatoes names
    with open(tomatoes_path, 'r') as f:
      tomatoes_names = f.readlines()

    self.n_tomatoes = len(tomatoes_names)

    # append real_data to X and y
    X.extend(tomatoes_names)
    y.extend([1]*self.n_tomatoes)

    self.n_samples = len(y)

    self.X = np.array(X)
    self.y = np.array(y)

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
        idx = idx.tolist()

    img_name = self.X[idx]

    
    sample = Image.open(img_name.rstrip())    
    sample = sample.resize((300, 300))
    
    sample = np.array(sample)
    
    transformation = self.transformations()
   
    return [transformation(sample).view((3, sample.shape[0], sample.shape[1])), self.y[idx]]

  def show(self, idx):

    if torch.is_tensor(idx):
        idx = idx.tolist()

    img_name = self.X[idx]
    label = self.y[idx]

    
    sample = Image.open(img_name.rstrip())    
    sample = sample.resize((64, 64))
    
    sample = np.array(sample)

    plt.imshow(sample)
    plt.axis('off')
    plt.title("Image number {} (label={})".format(idx, label))
    plt.show()

  def transformations(self):
    data_transform = transforms.Compose([transforms.ToTensor()])
    
    return data_transform

  def __len__(self):
    return self.n_samples

  def count(self):
    print("Number of real samples: {}\nNumber of fake samples: {}\nTOTAL: {}\n".format(self.n_real, self.n_fake, self.n_samples))

  
#%%DATA LOADER
  
def get_data_loaders():
    
    
    dataset = DATASET()
    index = random.randint(0, len(dataset)-1)
    dataset.show(index)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_size = int(0.8 * len(_dataset))
    val_size = len(_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(_dataset, [train_size, val_size])
    
    # Parameters
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 0}
      
      
    train_loader = data.DataLoader(train_dataset, **params)
    valid_loader = data.DataLoader(val_dataset, **params)
    test_loader = data.DataLoader(test_dataset, **params)
    
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    
    train, val, test = get_data_loaders()