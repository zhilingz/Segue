'''
1. train generator and surrogate model
2. save generator
'''
import sys
import os
import time
import torch
import torchvision
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0) # 固定随机种子，方便做对照试验
torch.cuda.manual_seed(0)

from dataset import getdataloader
from method import Method


# train setting
epochs = 30
save_epoch = 10 # 多少epoch保存一次
Epsilon = 8/255 # 8/255
# False: Supervised Scenario   True: Unupervised Scenario.
kmeans_label = True if sys.argv[1]=='True' else False 
# dataset setting
root = './'
datasetname = 'WebFace10'  
modelname =  'resnet18' 
img_size = 224
batch_size = 16
rho = 1

train_dataloader, test_dataloader = getdataloader(root, datasetname, img_size, batch_size)
# unlearnable method
num_classes = len(train_dataloader.dataset.classes)
method = Method( train_dataloader, test_dataloader, datasetname, modelname, num_classes, img_size, 
               Epsilon, kmeans_label, rho, device) 
# surrogate_model setting
surrogate_model = torchvision.models.resnet18(num_classes=num_classes).to(device)
if img_size <= 112:
    surrogate_model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    surrogate_model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).to(device)

optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=5e-4) #5e-4
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max =  epochs)
print('surrogate model:', modelname)

method.model_eval(surrogate_model, 0)

for epoch in range(1, epochs+1):
    method.loss_init()

    # train surrogate model
    if epoch%5 == 1: 
        surrogate_model.train()
        for step, data in enumerate(train_dataloader):
            image, label = data
            image, label = image.to(device), label.to(device)
            if method.kmeans_label: # 使用k-means预测的label
                with torch.no_grad():
                    features = method.netG.facenet.feature_extract(image).detach()
                    lab = method.netG.kmeans.predict(features.cpu())
                label = torch.from_numpy(lab).cuda().long()
            per_image = method.add_noise(image, label)
            logits = surrogate_model(per_image)
            loss_model = F.cross_entropy(logits, label)
            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()
            scheduler.step()
        method.model_eval(surrogate_model, epoch)
        time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        print(time_now, "epoch %d: lr_model:%.3e loss_model:%.2f"%(epoch, optimizer.param_groups[0]['lr'], loss_model))
    
    # train generator
    for _, data in enumerate(train_dataloader):
        image, label = data
        image, label = image.to(device), label.to(device)
        if method.kmeans_label: # 使用k-means预测的label
            with torch.no_grad():
                features = method.netG.facenet.feature_extract(image).detach()
                lab = method.netG.kmeans.predict(features.cpu())
            label = torch.from_numpy(lab).cuda().long()
        method.train_noise(surrogate_model, image, label)
    method.print_loss(epoch) # print statistics
    
    # save
    if loss_model < 0.01:
        method.model_eval(surrogate_model, epoch)
        method.save_G()
        break
    if epoch%save_epoch == 0: # 保存
        method.save_G()