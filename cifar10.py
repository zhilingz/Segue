import os
import shutil
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

cifar10_transforms = transforms.Compose([
    transforms.ToTensor(),
    ])
datasetname = 'CIFAR10'
root = '/data/zhangzhiling/'+datasetname
output_train = root+'/train_clean/'
output_test = root+'/test_clean/'
batch_size = 100
train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=cifar10_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
test_dataset = torchvision.datasets.CIFAR10(root, train=False, transform=cifar10_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
num_classes = len(train_dataset.classes)
rjust = lambda x:str(x).rjust(1,'0') # input:int, output:str
if not os.path.exists(output_train):
    os.mkdir(output_train)
    for i in range(num_classes):
        os.mkdir(output_train+rjust(i))
if not os.path.exists(output_test):
    os.mkdir(output_test)
    for i in range(num_classes):
        os.mkdir(output_test+rjust(i))
id_arr = [0 for _ in range(num_classes)]
for images, labels in tqdm(train_dataloader):
    for i, label in enumerate(labels):
        torchvision.utils.save_image(images[i],output_train+rjust(int(label))+'/'+str(id_arr[label])+'.jpg')
        id_arr[label]+=1
for images, labels in tqdm(test_dataloader):
    for i, label in enumerate(labels):
        torchvision.utils.save_image(images[i],output_test+rjust(int(label))+'/'+str(id_arr[label])+'.jpg')
        id_arr[label]+=1