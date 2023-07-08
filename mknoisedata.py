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

# if sys.argv[3] == 'log':
#     print_log = open("logs/"+sys.argv[4],"a",buffering=1)
#     sys.stdout = print_log
#     print(sys.argv)
logpath = '/data/zhangzhiling/Segue/log/'+'_'.join([sys.argv[7],sys.argv[5],sys.argv[2],sys.argv[1],sys.argv[6]])
if not os.path.exists(logpath):
    os.mkdir(logpath)
if sys.argv[3] == 'log':
    # print_log = open("logs/"+sys.argv[4],"a",buffering=1)
    # sys.stdout = print_log
    # print("\n\n", sys.argv)
    logfilepath = logpath+'/log.txt'
    print_log = open(logfilepath,"a",buffering=1)
    sys.stdout = print_log

for noise_prop in range(100,101):
    Epsilon = 8 * 1 / 255
    num_workers = 4
    noise_prop /= 100
    protect = 50 # 多少个类加噪声，其余类不加噪声
    bs1 = 10
    bs2 = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0) # 固定随机种子，方便做对照试验
    torch.cuda.manual_seed(0)

    method = sys.argv[1] # UEc UE RUE GUE random
    datasetname = sys.argv[5] # sys.argv[2] # WebFace10 WebFace10_ ImageNet10 CIFAR10 CIFAR10_0.2 CelebA10 VGGFace10
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
    test_dataset =torchvision.datasets.ImageFolder(root+'/test_clean',transform)
    test_sampler = CustomSampler(len(test_dataset))
    test_dataloader = DataLoader(test_dataset,batch_size=bs2,shuffle=False,sampler=test_sampler,num_workers=num_workers,drop_last=False)
    num_classes = len(train_dataset.classes) # 类别数
    num_samples = len(train_dataset)
    per_samples = num_samples/num_classes
    id_arr = [0 for _ in range(num_classes)]
    rjust = lambda x:str(x).rjust(len(str(num_classes-1)),'0') # input:int, output:str, 如果不补全则会出现label和文件名对不上的情况

    if method == 'random': # 随机噪声
        noise = torch.FloatTensor(*[len(train_dataset),3,resize,resize]).uniform_(-Epsilon,Epsilon).to(device)
        print('random noise generated!')
    if method == 'GUE':
        Eps = '4' # 对抗训练的强度
        # generator_path = 'ul_models/'+datasetname+'_GUE'+str(resize)+'a'+Eps+'.pth'
        generator_path ='ul_models/'+datasetname+method+'.pth' #'WebFace10'
        # generator_path = '/data/zhangzhiling/Segue/log/2023-07-01 23:22:56_WebFace10_mobilenet_v2_GUE_new/checkpoints/64.pth'
        netG = Generator(3,3,True,num_classes,bin_label=True,kmeans_label=False,datasetname=datasetname).to(device)
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
    if method == 'UEc' or method == 'UE' or method == 'TUE':
        # noise_path = 'ul_models/'+datasetname+'_'+method+str(resize)+'.pt'
        noise_path = './ul_models/'+datasetname+method+'.pt'
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
        # output_train = root+'/train'+str(noise_prop)+method+str(resize)+'a'+Eps+'/' # '+str(noise_prop)+' random _png '+str(resize)+'
        # output_test = root+'/test'+str(noise_prop)+method+str(resize)+'a'+Eps+'/'
    # output_trainclean = root+'/train_cleanjpg/'
    # output_testclean = root+'/test_cleanjpg/'
    output_train = root+'/train'+method+'/'
    output_test = root+'/test'+method+'/'
    for path in [output_train,output_test]:
        if not os.path.exists(path):
            os.mkdir(path)
            for i in range(num_classes):
                os.mkdir(path+rjust(i))
    # 生成加噪训练集
    idx = 0
    idxlist1300 = torch.zeros(1300)

    # import time
    # start_time = time.time()
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
        if method == 'UE' or method == 'RUE' or method == 'TUE' or method == 'random':
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
        for i, label in enumerate(labels):
            if i%10<noise_prop*10 and label<protect: # 不同毒化率：加噪
                path = output_train+rjust(int(label))+'/n'+str(id_arr[label])+extension
                ndarr = adv_images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(path, quality=quality)
                # path = output_trainclean+rjust(int(label))+'/'+str(id_arr[label])+extension
                # ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                # im = Image.fromarray(ndarr)
                # im.save(path, quality=quality)
                id_arr[label]+=1
            else: # 不同毒化率：不加噪
                #torchvision.utils.save_image(images[i],output_train+rjust(int(label))+'/'+str(id_arr[label])+extension) 
                path = output_train+rjust(int(label))+'/'+str(id_arr[label])+extension
                ndarr = adv_images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(path, quality=quality)
                id_arr[label]+=1
    print(output_train+' generated!')
    # end_time = time.time()
    # print('time cost : %.5f sec' %(start_time-end_time))
    # 生成加噪测试集
    idx = 0
    for images, labels in test_dataloader: # tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device)
        if method == 'GUE':
            perturbation = netG(images, labels)
        if method == 'UEc':
            batch_noise = [] 
            for label in labels:
                class_noise = noise[label]
                batch_noise.append(class_noise)
            perturbation = torch.stack(batch_noise).to(device)
        if method == 'UE'or method == 'RUE' or method == 'TUE' or method == 'random':
            perturbation = noise[idx:idx+len(labels)]
            idx += len(labels)
        perturbation = torch.clamp(perturbation, -Epsilon, Epsilon)
        adv_images = perturbation + images
        adv_images = torch.clamp(adv_images, 0, 1)
        for i, label in enumerate(labels):
            if i%10<noise_prop*10 and label<protect: # shuffle=false时，不同毒化率：加噪
                #torchvision.utils.save_image(adv_images[i],output_test+rjust(int(label))+'/n'+str(id_arr[label])+extension)
                path = output_test+rjust(int(label))+'/n'+str(id_arr[label])+extension
                ndarr = adv_images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(path, quality=quality)
                # path = output_testclean+rjust(int(label))+'/'+str(id_arr[label])+extension
                # ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                # im = Image.fromarray(ndarr)
                # im.save(path, quality=quality)
                id_arr[label]+=1
            else:
                # torchvision.utils.save_image(images[i],output_test+rjust(int(label))+'/'+str(id_arr[label])+extension) 
                path = output_test+rjust(int(label))+'/'+str(id_arr[label])+extension
                ndarr = adv_images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(path, quality=quality)
                id_arr[label]+=1
    print(output_test+' generated!')

