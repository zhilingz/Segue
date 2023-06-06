import os
import sys
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import lpips
import numpy as np
from pytorch_msssim import ssim

from models import Generator
from dataset import CustomSampler

if sys.argv[3] == 'log':
    # savedStdout = sys.stdout  #保存标准输出流
    print_log = open("logs/iqalog.txt","a",buffering=1)
    sys.stdout = print_log
    print(sys.argv)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0) # 固定随机种子，方便做对照试验
torch.cuda.manual_seed(0)

Epsilon = 8/255
size = 224
method = sys.argv[1]
datasetname = sys.argv[2]
root = '/data/zhangzhiling/'+datasetname
train_transforms = transforms.Compose([
    transforms.Resize([size,size]),
    transforms.ToTensor(),
    ])
train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',train_transforms)
train_sampler = CustomSampler(len(train_dataset))
train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=False,sampler=train_sampler,num_workers=4,drop_last=True)
num_classes = len(train_dataset.classes)
lpips_loss = lpips.LPIPS().to(device) # net='vgg'
if method == 'GUE':
    Eps = '4' # 对抗训练的强度
    generator_path ='ul_models/'+datasetname+method+'.pth'
    netG = Generator(3,3,True,num_classes,bin_label=False,kmeans_label=False,datasetname=datasetname).to(device)
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
    # netG.load_state_dict(torch.load(generator_path))
    netG.eval()
if method == 'random':
    noise = torch.FloatTensor(*[len(train_dataset),3,size,size]).uniform_(-Epsilon,Epsilon).to(device)
if method == 'UEc' or method == 'UEs' or method == 'TUE' or method == 'RUE':
    noise_path = './ul_models/'+datasetname+method+'.pt'
    noise = torch.load(noise_path)
# if method == 'RUE':
#     Eps = '4' # 对抗训练的强度
#     noise_path = './ul_models/'+datasetname+'RUE.pt'
#     noise = torch.load(noise_path) # 在GPU上生成，load到GPU
#     print(noise_path, 'load!')
idx = 0
mse_sum = 0
lpips_sum = 0
ssim_sum = 0
train_itr = 0
with torch.no_grad():
    for _, data in enumerate(train_dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        if method == 'GUE':
            perturbation = netG(images, labels)
        if method == 'UEc':
            batch_noise = [] 
            for label in labels:
                class_noise = noise[label]
                batch_noise.append(class_noise)
            perturbation = torch.stack(batch_noise).to(device)
        if method == 'UEs' or method == 'RUE' or method == 'TUE' or method == 'random':
            perturbation = noise[idx:idx+len(labels)]
            idx += len(labels)
        perturbation = torch.clamp(perturbation, -Epsilon, Epsilon)
        adv_images = perturbation + images
        adv_images = torch.clamp(adv_images, 0, 1)
        mse_sum += F.mse_loss(images,adv_images)
        lpips_sum += lpips_loss.forward(images,adv_images)
        ssim_sum += ssim(images,adv_images,data_range=1.0)
        train_itr += 1
print('MSE: %.6f\nLPIPS: %.6f\nssim: %.6f'%(
    mse_sum/train_itr, lpips_sum.mean()/train_itr, ssim_sum/train_itr))
