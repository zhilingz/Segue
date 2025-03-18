import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

def getdataloader(root, datasetname, size, batch_size):
    train_transforms = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        ])
    root = root+datasetname
    train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',train_transforms)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=False)
    test_dataset = torchvision.datasets.ImageFolder(root+'/test_clean',test_transforms)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=False)

    print('train_dataset:\n',train_dataset.root,'\n',train_dataset.transform)
    print('test_dataset:\n',test_dataset.root,'\n',test_dataset.transform)
    return train_dataloader, test_dataloader