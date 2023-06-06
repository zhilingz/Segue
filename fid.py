import os
import sys
from openpyxl import Workbook
import time 
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

method = sys.argv[1]
datasetname = sys.argv[2]

batch_size = 10
train_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
])
root = '/data/zhangzhiling/'+datasetname
train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',train_transforms)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=4,drop_last=False)
train_dataset_noise = torchvision.datasets.ImageFolder(root+'/train'+method,train_transforms) # +'_png' str(noise_prop) +str(resize)
train_dataloader_noise = DataLoader(train_dataset_noise,batch_size=batch_size,shuffle=False,num_workers=4,drop_last=False) 
output_clean = root+'/fidtrainclean/'
output_noise = root+'/fidtrain'+method+'/'
num_classes = len(train_dataset.classes)
id_arr = [0 for _ in range(num_classes)]
rjust = lambda x:str(x).rjust(len(str(num_classes-1)),'0')
if not os.path.exists(output_clean):
    os.mkdir(output_clean)
    for images, labels in train_dataloader:
        for i, label in enumerate(labels):
            torchvision.utils.save_image(images[i],output_clean+rjust(int(label))+'_'+str(id_arr[label])+'.png') 
            id_arr[label]+=1
if not os.path.exists(output_noise):
    os.mkdir(output_noise)
    id_arr = [0 for _ in range(num_classes)]
    for images, labels in train_dataloader_noise:
        for i, label in enumerate(labels):
            torchvision.utils.save_image(images[i],output_noise+rjust(int(label))+'_'+str(id_arr[label])+'.png') 
            id_arr[label]+=1
