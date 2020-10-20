
from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import scipy.io as scio
import torch
from scipy import ndimage, misc

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class McDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
       
        imgs= os.listdir(self.image_dir)
        self.A_paths = []
        self.A_names = []
        for img in imgs :
            imgpath=os.path.join(self.image_dir,img)
            self.A_paths.append(imgpath)
            self.A_names.append(img.split('.')[0])
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        #print("read meta done")
        self.initialized = False
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)

               
        A_name = self.A_names[index % self.A_size]
        return {'A': A, 'path': A_path, 'name': A_name }
