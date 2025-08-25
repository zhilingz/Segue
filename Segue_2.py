'''
1. load generator
2. generate perturbations for iamges and save perturbed images (to form unlearnable dataset)
'''
import sys
import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from models import Generator
from tqdm import tqdm

Epsilon = 8 / 255
num_workers = 4
bs1 = 10
bs2 = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

datasetname = 'WebFace10' 
root = './'+datasetname
resize = 32 if 'CIFAR10' in datasetname else 224

# False: Supervised Scenario   True: Unupervised Scenario.
kmeans_label = True if sys.argv[1]=='True' else False 
print("kmeans_label",kmeans_label)
extension = '.png'
transform = transforms.Compose([
    # transforms.CenterCrop(150),
    transforms.Resize([resize,resize]),
    transforms.ToTensor(),
    ])
train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',transform)
train_dataloader = DataLoader(train_dataset,batch_size=bs1,shuffle=False,num_workers=num_workers,drop_last=False)
test_dataset =torchvision.datasets.ImageFolder(root+'/test_clean',transform)
test_dataloader = DataLoader(test_dataset,batch_size=bs2,shuffle=False,num_workers=num_workers,drop_last=False)
num_classes = len(train_dataset.classes)
num_samples = len(train_dataset)
per_samples = num_samples/num_classes
id_arr = [0 for _ in range(num_classes)]

# Without this completion, it will cause a mismatch between the labels and the filenames.
# input:int, output:str
rjust = lambda x:str(x).rjust(len(str(num_classes-1)),'0')

# 1. load generator
generator_path ='ul_models/'+datasetname+'_G.pth' #'WebFace10'
netG = Generator(3,3,num_classes,kmeans_label=kmeans_label,datasetname=datasetname).to(device)
model_dict = netG.state_dict()
pretrained_dict = torch.load(generator_path)
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
netG.load_state_dict(model_dict)
print(generator_path, 'load!')
netG.eval()
output_train = root+'/train_noise/'
output_test = root+'/test_noise/'
for path in [output_train,output_test]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for i in range(num_classes):
        os.mkdir(path+rjust(i))

#  2. generate perturbations for iamges and save 
for images, labels in tqdm(train_dataloader):
    images, labels = images.to(device), labels.to(device)
    perturbation = netG(images, labels)
    perturbation = torch.clamp(perturbation, -Epsilon, Epsilon)
    adv_images = perturbation + images
    adv_images = torch.clamp(adv_images, 0, 1)
    for i, label in enumerate(labels):
        path = output_train+rjust(int(label))+'/n'+str(id_arr[label])+extension
        torchvision.utils.save_image(adv_images[i],path)
        id_arr[label]+=1
print(output_train+' generated!')

for images, labels in tqdm(test_dataloader):
    images, labels = images.to(device), labels.to(device)
    perturbation = netG(images, labels)
    perturbation = torch.clamp(perturbation, -Epsilon, Epsilon)
    adv_images = perturbation + images
    adv_images = torch.clamp(adv_images, 0, 1)
    for i, label in enumerate(labels):
        path = output_test+rjust(int(label))+'/n'+str(id_arr[label])+extension
        torchvision.utils.save_image(adv_images[i],path)
        id_arr[label]+=1
print(output_test+' generated!')
