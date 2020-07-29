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
    
PATH = "/ctm-hdd-pool01/DB/LivDet2015/train/"

