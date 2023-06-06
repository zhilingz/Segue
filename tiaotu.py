import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from models import Generator
import pickle
from PIL import Image
from tqdm import tqdm
from dataset import CustomSampler

for method in ["UEs", "GUE", "TUE", "RUE"]:
    Epsilon = 8 * 1 / 255
    num_workers = 4
    noise_prop = 1
    protect = 50 # 多少个类加噪声，其余类不加噪声
    bs1 = 100
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0) # 固定随机种子，方便做对照试验
    torch.cuda.manual_seed(0)

    # UEc UEs RUE TUE GUE random
    datasetname = "WebFace10" #'WebFace10' # WebFace10 WebFace10_ ImageNet10 CIFAR10 CIFAR10_0.2 CelebA10 VGGFace10
    root = '/data/zhangzhiling/'+datasetname
    resize = 32 if 'CIFAR10' in datasetname else 224
    quality =  95 # jgep 压缩质量int(sys.argv[2])
    extension = '.png'
    transform = transforms.Compose([
        # transforms.CenterCrop(150),
        transforms.Resize([resize,resize]),
        transforms.ToTensor(),
        ])
    train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',transform)
    train_sampler = CustomSampler(len(train_dataset))
    train_dataloader = DataLoader(train_dataset,batch_size=bs1,shuffle=False,sampler=train_sampler,num_workers=num_workers,drop_last=False)
    num_classes = len(train_dataset.classes) # 类别数
    num_samples = len(train_dataset)
    per_samples = num_samples/num_classes
    id_arr = [0 for _ in range(num_classes)]

    if method == 'random': # 随机噪声
        noise = torch.FloatTensor(*[len(train_dataset),3,resize,resize]).uniform_(-Epsilon,Epsilon).to(device)
        print('random noise generated!')
    if method == 'GUE':
        generator_path ='ul_models/'+datasetname+method+'.pth' #'WebFace10'
        netG = Generator(3,3,True,num_classes,bin_label=True,kmeans_label=True,datasetname=datasetname).to(device)
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
        print(generator_path, 'load!')
        netG.eval()
        # output_train = root+'/train'+str(noise_prop)+method+str(resize)+'t10/'# 'a'+Eps+'/' # '+str(noise_prop)+' random _png '+str(resize)+'
        # output_test = root+'/test'+str(noise_prop)+method+str(resize)+'t10/'# a'+Eps+'/'
    if method == 'UEc' or method == 'UEs' or method == 'TUE':
        # noise_path = 'ul_models/'+datasetname+'_'+method+str(resize)+'.pt'
        noise_path = './ul_models/'+'WebFace10'+method+'.pt'
        noise = torch.load(noise_path)
        print(noise_path, 'load!')
        # output_train = root+'/train'+str(noise_prop)+method+str(resize)+'/' # '+str(noise_prop)+' random _png '+str(resize)+'
        # output_test = root+'/test'+str(noise_prop)+method+str(resize)+'/'
    if method == 'RUE':
        Eps = '4' # 对抗训练的强度
        # noise_path = 'ul_models/'+datasetname+'_RUE'+str(resize)+'a'+Eps+'.pt'
        noise_path = './ul_models/'+datasetname+'RUE.pt'
        noise = torch.load(noise_path) # 在GPU上生成，load到GPU
        print(noise_path, 'load!')

    # 生成加噪训练集
    idx = 0
    idxlist1300 = torch.zeros(1300)

    for images, labels in train_dataloader:# tqdm(train_dataloader):
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
            if noise.shape[0] == num_samples:
                perturbation = noise[idx:idx+len(labels)]
                idx += len(labels)
            else:
                batch_noise = []
                for idx200 in range(idx,idx+len(labels)):
                    idx_indices200 = train_sampler.indices200[idx200]
                    idx_indices1300 = int((idx_indices200//per_samples)*(noise.shape[0]//num_classes)+idx_indices200%per_samples)
                    idxlist1300[idx_indices1300]+=1
                    batch_noise.append(noise[train_sampler.indices1300.index(idx_indices1300)])
                perturbation = torch.stack(batch_noise).to(device)
                idx += len(labels)
        perturbation = torch.clamp(perturbation, -Epsilon, Epsilon)
        adv_images = perturbation + images
        adv_images = torch.clamp(adv_images, 0, 1)
        
        batch_clean = []
        batch_adv_images = []
        batch_perturbation = []
        for i, label in enumerate(labels):
            if label==0:
                batch_adv_images.append(adv_images[i])
                batch_perturbation.append(perturbation[i])
                batch_clean.append(images[i])
            if len(batch_adv_images)>=2:
                break
        for i, label in enumerate(labels):
            if label==1:
                batch_adv_images.append(adv_images[i])
                batch_perturbation.append(perturbation[i])
                batch_clean.append(images[i])
            if len(batch_adv_images)>=4:
                break
        for i, label in enumerate(labels):
            if label==4:
                batch_adv_images.append(adv_images[i])
                batch_perturbation.append(perturbation[i])
                batch_clean.append(images[i])
            if len(batch_adv_images)>=6:
                break
        torchvision.utils.save_image(torch.stack(batch_adv_images),'images/adv_'+method+'.png',nrow=1) 
        torchvision.utils.save_image(torch.stack(batch_perturbation)*100,'images/per_'+method+'.png',nrow=1) 
        torchvision.utils.save_image(torch.stack(batch_clean),'images/clean.png',nrow=1) 
        print(method+'finish')
        break

       