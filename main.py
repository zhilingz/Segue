import os
import time 
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from models import Facenet, Generator
from attack import ul_Attack

for Epsilon in range(8,9):
    train = True
    test = False
    lr1 = 5e-4
    lr2 = 5e-4 # 5e-4
    epochs_train = 21
    epochs_test = 200
    batch_size = 16
    resize = 224
    Epsilon = Epsilon * 1 / 255
    target_class = False  # False表示全类别保护；True表示做十组实验，每组实验只保护一类图片
    noise_prop = 1 # 0到1，训练集中加噪声图片的比例
    data_prop = 1 # 0到1，训练集每个类中数据的使用比例
    label_feature = True # 将类别信息加到图片特征中
    face_label = False # 使用预训练的人脸提取器和arcface得到预测标签，代替真实标签嵌入图片特征中
    kmeans_label = False # 使用预训练的人脸提取器提取人脸128维特征，用kmeans聚类预测标签代替真实标签
    gradient_base = False # 基于梯度的方法
    class_wise = True # 类噪声（或者样本噪声）
    datasetname = 'WebFace10'  # CIFAR10 WebFace10 ImageNet10
    generator_path = 'ul_models/Face_e8_netG_truelabel.pth'
    noise_path = './ul_models/sample_perturbation_epoch_20.pt'
    target_model_file = './ul_models/target_model_trained_with_clean_30.pth'
    image_nc = 3 # 通道数，RGB为3
    num_workers = 4
    num_classes = 10 # 类别数量
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0) # 固定随机种子，方便做对照试验
    torch.cuda.manual_seed(0)

    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    # print("CUDA Available:",torch.cuda.is_available())
    print('gradient_base: {}\nlabel_feature: {}\nclass_wise: {}\nnoise_prop: {}\ndata_prop: {}\nEpsilon: {:.4f}={}/255\nlr1: {}\nlr2: {}'.format(
            gradient_base, label_feature, class_wise, noise_prop, data_prop, Epsilon, Epsilon*255, lr1, lr2))

    train_transforms = transforms.Compose([
        # transforms.RandomCrop(250, padding=40),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.RandomGrayscale(p=0.1),
        transforms.Resize([resize,resize]),
        transforms.ToTensor(),
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([resize,resize]),
        transforms.ToTensor(),
        ])
    cifar10_transforms = transforms.Compose([
        transforms.ToTensor(),
        ])

    class CIFAR10_Mini(torchvision.datasets.CIFAR10):
        '''只取数据集的20%'''
        def __len__(self) -> int:
            return int(len(self.data)*data_prop)  
    class CIFAR10_Target(torchvision.datasets.CIFAR10):
        def __getitem__(self, index):  
            img, target = self.data[index], self.targets[index]
            # 若搜到数据集最后一个不是target图片，则使用第一个target图片
            if index==self.__len__()-1 and target!=target_class:
                return self.__getitem__(4)
            if target!=target_class: # 将非target图片替代为下一个target图片
                return self.__getitem__(index+1)
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    root = '/data/zhangzhiling/'+datasetname
    if datasetname == 'CIFAR10':
        train_dataset = CIFAR10_Mini(root, train=True, transform=cifar10_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        test_dataset = CIFAR10_Mini(root, train=False, transform=cifar10_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        # 将要攻击的目标模型，此模型是分类模型（训练集50000，测试集10000）
        # 更新生成器时不更新目标模型的参数，使得其生成的图片对于目标模型的loss尽可能小
        target_model = torchvision.models.resnet18(num_classes=num_classes).to(device)
        target_model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        target_model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).to(device)
    if datasetname == 'WebFace10' or datasetname == 'ImageNet10':
        train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',train_transforms)
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
        test_dataset =torchvision.datasets.ImageFolder(root+'/test_clean',test_transforms)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
        target_model = torchvision.models.resnet18(num_classes=len(train_dataset.classes)).to(device)
        # target_model = Facenet(num_classes=len(train_dataset.classes)).to(device)
    if os.path.exists(target_model_file):
        target_model.load_state_dict(torch.load(target_model_file))
        print(target_model_file+' load!')
    # target_model.load_state_dict(torch.load(pretrained_model, map_location=device))
    else: # 如果代理模型地址不存在，则初始化一个代理模型
        print(target_model_file+' init!')

    ul_attack=ul_Attack(device,target_model,image_nc,Epsilon,lr1,lr2,target_class,
                    gradient_base,class_wise, noise_prop,label_feature,
                    train_dataloader,test_dataloader,epochs_train,epochs_test,
                    generator_path,noise_path,face_label,kmeans_label)

    if train == True:
        print('Noise training start')
        ul_attack.train()
        print('Noise training finish')
    if test == True:
        print('Noise testing start')
        ul_attack.test()
        print('Noise testing finish')