import os
import time 
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import Facenet

def _model_eval(set=['clean','trainset']):
    '''测试代理模型在加噪/干净的训练/测试数据集上的准确率'''
    # num_samples = train_sum if set[1]=='trainset' else test_sum
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
        _model_eval(['dirty','trainset'])
        _model_eval(['dirty','testset '])
        _model_eval(['clean','trainset'])
        _model_eval(['clean','testset '])
    
def model_load(model,modelname):
    model_path = 'pre_models/facenet_'+modelname+'.pth'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    return model

def TNSE_view(train_dataloader,model):
    num_samples = len(train_dataloader.dataset.targets)
    num_features = model.last_bn.num_features
    num_classes = model.num_classes
    feature = torch.zeros(num_samples,num_features)
    labels = torch.zeros(num_samples)
    plt.figure(figsize=(12, 12))
    tsne = TSNE(n_components=2, perplexity=num_samples/num_classes,init='pca',learning_rate='auto')
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            image, label = data
            feature[i:len(label)+i] = model.feature_extract(image.to(device))
            labels[i:len(label)+i] = label
    data_2d = tsne.fit_transform(feature.view(num_samples,-1))
    for i in range(num_classes):
        plt.scatter(data_2d[labels == i, 0], data_2d[labels == i, 1], c=colours[i], label=i, alpha=0.1)
    plt.legend()
    plt.savefig('images/tnse.jpg')

for noise_prop in range(5,6): 
    lr1 = 5e-4
    lr2 = 5e-4 # 5e-4
    epochs_test = 80
    batch_size = 16
    resize = 224
    num_workers = 4
    target_class = False  # False表示全类别保护；True表示做十组实验，每组实验只保护一类图片
    noise_prop /= 5 # 0到1，训练集中加噪声图片的比例
    data_prop = 1 # 0到1，训练集每个类中数据的使用比例
    label_feature = True # 将类别信息加到图片特征中
    gradient_base = False # 基于梯度的方法
    class_wise = False # 类噪声（或者样本噪声）
    tensorboard = False # 启用tensorboard
    modelname = 'mobilenet' # resnet18 mobilenet inception_resnetv1
    datasetname = 'WebFace10' # WebFace10 ImageNet10 CIFAR10
    pretrain_model = True

    torch.manual_seed(0) # 固定随机种子，方便做对照试验
    torch.cuda.manual_seed(0)

    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    # print("CUDA Available:",torch.cuda.is_available())
    print('gradient_base: {}\nlabel_feature: {}\nclass_wise: {}\nnoise_prop: {}\ndata_prop: {}\nlr1: {}\nlr2: {}'.format(
            gradient_base, label_feature, class_wise, noise_prop, data_prop, lr1, lr2))

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
    
    root = 'dataset/'+datasetname
    train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',test_trasforms)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False)
    test_dataset = torchvision.datasets.ImageFolder(root+'/test_clean',test_trasforms)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) # random
    train_dataset_noise = torchvision.datasets.ImageFolder(root+'/train_noise1.0',train_transforms)
    train_dataloader_noise = DataLoader(train_dataset_noise,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) 
    test_dataset_noise = torchvision.datasets.ImageFolder(root+'/test_noise1.0',test_trasforms)
    test_dataloader_noise = DataLoader(test_dataset_noise,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False)
    print('train_dataset:',train_dataset.root)
    print('test_dataset:',test_dataset.root)
    print('train_dataset_noise:',train_dataset_noise.root)
    print('test_dataset_noise:',test_dataset_noise.root)
    # model = torchvision.models.resnet18(num_classes=len(train_dataset.classes)).to(device)
    if modelname=='mobilenet' or modelname=='inception_resnetv1':
        model = Facenet(backbone=modelname, num_classes=len(train_dataset.classes)).to(device)  
        loss_fun = lambda logits, labels : torch.nn.NLLLoss()(F.log_softmax(logits, dim = -1), labels)  
        if pretrain_model:
            model = model_load(model,modelname)
    elif modelname=='resnet18':
        model = torchvision.models.resnet18(num_classes=len(train_dataset.classes)).to(device)
        loss_fun = lambda logits, labels : F.cross_entropy(logits, labels)
        # model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        # model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).to(device)
    
    train_itr = len(train_dataloader)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr2)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_model, T_max = epochs_test)
    print('modelname: ',modelname)

    model_eval(0)
    for epoch in range(1, epochs_test+1):
        loss_sum = 0
        model.train()
        for _, data in enumerate(train_dataloader):
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
        print(time_now, 'Loss in Epoch %d: %.4f' % (epoch, loss_sum/train_itr))
        if epoch %2 == 1:
            print('lr:',optimizer_model.param_groups[0]['lr']) # 打印学习率
            model_eval(epoch)
        # if epoch%10 == 0:
        #     TNSE_view(train_dataloader,model)
        if loss_sum/train_itr<0.001:
            break
        if epoch%5==0:# 保存
            netG_file_name = './ul_models/facenet_Webface10.pth'
            torch.save(model.state_dict(), netG_file_name)
            print(netG_file_name,'saved!')
