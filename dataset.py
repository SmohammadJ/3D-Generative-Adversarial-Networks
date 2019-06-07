from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from torchvision import datasets, transforms
import torch.nn as nn
import torch
import gzip
from mnist import MNIST
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Minst3D (Dataset):
    def __init__(self, root_dir=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        mndata = MNIST('mnist_data')

        self.images, _ = mndata.load_training()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        array = np.array(self.images[idx]).reshape(28,28)
        image = Image.fromarray(array)
        
        if self.transform:
            image = self.transform( image )

        voxel = torch.zeros((1,32,32,32))
        voxel[:,:,:,:10] = torch.stack([image]*10, dim=3)
        
        return(voxel, idx)
        



    



    
