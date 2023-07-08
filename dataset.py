import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

class CustomSampler(Sampler):
    '''
    保证每个epoch的sample序列是一样的
    '''
    def __init__(self, num_samples):
        self.num_samples = num_samples
        # self.indices200 = torch.load('ul_models/indices200.pt').tolist()
        self.indices1300 = torch.load('ul_models/indices1300.pt').tolist()
        self.indices200 = torch.load('ul_models/indices200.pt').tolist()
        if num_samples == 200:
            self.indices = self.indices200
        elif num_samples == 1300:
            self.indices = self.indices1300
        else:
            self.indices = torch.randperm(self.num_samples)
        # self.indices = torch.randperm(self.num_samples)
        # if num_samples == 200 or num_samples == 1300:
        #     self.indices = torch.load('ul_models/indices'+str(num_samples)+'.pt')
        # else:
        #     self.indices = torch.randperm(self.num_samples)
    def __iter__(self):
        # torch.manual_seed(0)
        # indices = torch.arange(self.num_samples)
        return iter(self.indices)

    def __len__(self):
        return len(self.num_samples)

def getdataloader(root, datasetname, size, batch_size):
    train_transforms = transforms.Compose([
        # transforms.RandomCrop(250, padding=40),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.RandomGrayscale(p=0.1),
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        ])
    # if datasetname == 'CIFAR10': # 训练集50000，测试集10000
    #     train_dataset = torchvision.datasets.CIFAR10(root+datasetname, train=True, transform=train_transforms)
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    #     test_dataset = torchvision.datasets.CIFAR10(root+datasetname, train=False, transform=test_transforms)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # else:
    root = root+datasetname
    train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',train_transforms)
    train_sampler = CustomSampler(len(train_dataset))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,sampler=train_sampler,num_workers=4,drop_last=False)
    test_dataset = torchvision.datasets.ImageFolder(root+'/test_clean',test_transforms)
    test_sampler = CustomSampler(len(test_dataset))
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,sampler=test_sampler,num_workers=4,drop_last=False)

    print('train_dataset:\n',train_dataset.root,'\n',train_dataset.transform)
    print('test_dataset:\n',test_dataset.root,'\n',test_dataset.transform)
    return train_dataloader, test_dataloader