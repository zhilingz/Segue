'''
1. load clean dataset and unlearnable dataset
2. attack model
3. train attack model on unlearnable dataset and test on clean dataset
'''

import os
import time 
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def _model_eval(set=['clean','trainset']):
    '''测试代理模型在加噪/干净的训练/测试数据集上的准确率'''
    if set[0]=='dirty' and set[1]=='trainset':
        dataloader = train_dataloader_noise
    elif set[0]=='dirty' and set[1]=='testset ':
        dataloader = test_dataloader_noise
    elif set[0]=='clean' and set[1]=='trainset':
        dataloader = train_dataloader
    elif set[0]=='clean' and set[1]=='testset ':
        dataloader = test_dataloader
    num_classes = len(dataloader.dataset.classes)
    num_correct = torch.zeros(num_classes).to(device)
    num_samples = torch.zeros(num_classes).to(device)
    num_correct_sum = 0
    model.eval() # 停用dropout并在batchnorm层使用训练集的数据分布
    for _, data in enumerate(dataloader, 0):
        image, label = data
        image, label = image.to(device), label.to(device)
        pred_lab = torch.argmax(model(image),1)
        mask=pred_lab==label
        for i in range(len(label)):
            num_correct[label[i]]+=mask[i]
            num_samples[label[i]]+=1
        num_correct_sum += torch.sum(pred_lab==label,0)
    acc = num_correct/num_samples
    acc_sum = num_correct_sum/torch.sum(num_samples) 
    time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    print(time_now,'Agent model accuracy of {} {}: {} {:.2%}'.format(set[0],set[1],[round(i,2) for i in acc.tolist()],acc_sum))
    return acc_sum

def model_eval(epoch):
    with torch.no_grad():
        if tensorboard:
            writer.add_scalars('acc', {
                'dirty_trainset':_model_eval(['dirty','trainset']),
                'dirty_testset':_model_eval(['dirty','testset ']),
                'clean_trainset':_model_eval(['clean','trainset']),
                'clean_testset':_model_eval(['clean','testset '])
            }, epoch)
        else:
            _model_eval(['dirty','trainset'])
            _model_eval(['dirty','testset '])
            _model_eval(['clean','trainset'])
            _model_eval(['clean','testset '])
    
# def model_load(model,modelname):
#     model_path = 'pre_models/facenet_'+modelname+'.pth'
#     model_dict = model.state_dict()
#     pretrained_dict = torch.load(model_path, map_location = device)
#     load_key, no_load_key, temp_dict = [], [], {}
#     for k, v in pretrained_dict.items():
#         if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#             temp_dict[k] = v
#             load_key.append(k)
#         else:
#             no_load_key.append(k)
#     model_dict.update(temp_dict)
#     model.load_state_dict(model_dict)
#     print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
#     return model

lr = 5e-4 # 5e-4
epochs_test = 40
batch_size = 16
num_workers = 4
tensorboard = False # 启用tensorboard
modelname = 'resnet18'
datasetname = 'WebFace10' 
resize = 224

# adv_train = False
# atk_Epsilon = 4/255 
# atk_step_size = atk_Epsilon/5
# atk_step = 5
torch.manual_seed(0) # 固定随机种子，方便做对照试验
torch.cuda.manual_seed(0)

print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
# print("CUDA Available:",torch.cuda.is_available())
# print('adv_train: {}\natk_step: {}\natk_step_size: {}/255\natk_Epsilon: {}/255\n\nbatch_size: {}'.format(
#     adv_train,atk_step,atk_step_size*255,atk_Epsilon*255,batch_size))


# 1. load clean dataset and unlearnable dataset
train_transforms = transforms.Compose([
    transforms.Resize([resize,resize]),
    transforms.ToTensor(),
    ])
test_trasforms = transforms.Compose([
    transforms.Resize([resize,resize]),
    transforms.ToTensor(),
    ])
print('train_transforms:\n',train_transforms)
print('test_trasforms:\n',test_trasforms)
root = './'+datasetname
train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',train_transforms)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False)
test_dataset = torchvision.datasets.ImageFolder(root+'/test_clean',test_trasforms)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) 
train_dataset_noise = torchvision.datasets.ImageFolder(root+'/train_noise',train_transforms) 
train_dataloader_noise = DataLoader(train_dataset_noise,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) 
test_dataset_noise = torchvision.datasets.ImageFolder(root+'/test_noise',test_trasforms)
test_dataloader_noise = DataLoader(test_dataset_noise,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False)
print('train_dataset:',train_dataset.root)
print('test_dataset:',test_dataset.root)
print('train_dataset_noise:',train_dataset_noise.root)
print('test_dataset_noise:',test_dataset_noise.root)
num_classes=len(train_dataset.classes)

# 2. attack model
model = torchvision.models.resnet18(num_classes=num_classes).to(device)
if resize <= 112:
    model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).to(device)
loss_fun = lambda logits, labels : F.cross_entropy(logits, labels)


# 3. train attack model on unlearnable dataset and test on clean dataset
train_itr = len(train_dataloader)
optimizer_model = torch.optim.Adam(model.parameters(), lr=lr) # todo 不同分类器消融实验
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_model, T_max = epochs_test)
print('modelname:', modelname)
print('num_classes:', num_classes)
model_eval(0)
for epoch in range(1, epochs_test+1):
    loss_sum = 0
    model.train()
    for data in train_dataloader_noise: 
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_model = loss_fun(logits, labels)
        optimizer_model.zero_grad()
        loss_model.backward()
        optimizer_model.step()
        scheduler.step()
        loss_sum += loss_model
    time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    print(time_now, 'Loss in Epoch {}: {:.4f}'.format(epoch, loss_sum/train_itr))
    if epoch %5 == 1:
        time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        print(time_now, 'lr in Epoch {}:{:.3e}'.format(epoch, optimizer_model.param_groups[0]['lr'])) # 打印学习率
        model_eval(epoch)
    if loss_sum/train_itr<0.0001:
        model_eval(epoch)
        break