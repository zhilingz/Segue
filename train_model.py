import os
import sys
from openpyxl import Workbook
import time 
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import Facenet, Bottleneck, InvertedResidual
from util import AddPepperNoise, Cutout, mix_criterion, Mixup_data, Cutmix_data

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if sys.argv[3] == 'log':
    # savedStdout = sys.stdout  #保存标准输出流
    print_log = open("logs/"+sys.argv[4],"a",buffering=1) # printlog.txt
    sys.stdout = print_log
    print(sys.argv)

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

def get_advimg(x,y):
    adv_per = torch.zeros_like(x)
    adv_per.requires_grad_(True)
    for _ in range(atk_step): 
        logits_adv = model(adv_per+x)
        loss_adv = F.cross_entropy(logits_adv, y)
        grad_adv = torch.autograd.grad(loss_adv, [adv_per])[0]
        with torch.no_grad(): # loss最大化，对抗训练
            adv_per.add_(torch.sign(grad_adv), alpha=atk_step_size)
            adv_per.clamp_(-atk_Epsilon, atk_Epsilon)
    return ((adv_per+x).data).clamp_(0,1)

def check_size(train_dataset):
    train_image = Image.open(train_dataset.imgs[0][0])
    # test_image = Image.open(test_dataset.imgs[0][0])
    if train_image.size == (resize,resize):
        return True
    else:
        return False

def save_excel():
    '''保存至excel文件'''
    dataloader_list = [train_dataloader_noise,test_dataloader_noise,train_dataloader,test_dataloader]
    # 创建一个workbook并设置编码
    wb = Workbook()
    # 添加sheet
    ws = wb.active
    ws.title = 'acc of respective labels'
    for column in range(4):
        dataloader = dataloader_list[column]
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
        # acc_sum = num_correct_sum/torch.sum(num_samples) 
        for i, acc_it in enumerate(acc):    # 写入excel, 参数对应 行, 列, 值
            ws.cell(row=i+1, column=column+1).value = acc_it.item()
    # 保存
    wb.save('acc.xlsx')
    
for noise_prop in range(5,6): 
    lr = 5e-4
    epochs_test = 40
    batch_size = 16
    num_workers = 4
    target_class = False  # False表示全类别保护；True表示做十组实验，每组实验只保护一类图片
    noise_prop /= 5 # 0到1，训练集中加噪声图片的比例
    tensorboard = True # 启用tensorboard
    modelname = sys.argv[2] # # resnet18 mobilenet inception_resnetv1 sys.argv[2] 
    method = sys.argv[1] # UEc UEs RUE GUE random
    datasetname = 'WebFace10' # # WebFace10 WebFace50 WebFace10_ ImageNet10 CIFAR10 CIFAR10_0.2 CelebA10 VGGFace10
    
    resize = 32 if 'CIFAR10' in datasetname else 224
    pretrain_model = False
    adv_train = False
    atk_Epsilon = 0/255 #  int(sys.argv[2])/255 # 
    atk_step_size = atk_Epsilon/5
    atk_step = 5
    torch.manual_seed(0) # 固定随机种子，方便做对照试验
    torch.cuda.manual_seed(0)

    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    # print("CUDA Available:",torch.cuda.is_available())
    print('adv_train: {}\natk_step: {}\natk_step_size: {}/255\natk_Epsilon: {}/255\nnoise_prop: {}\nbatch_size: {}'.format(
        adv_train,atk_step,atk_step_size*255,atk_Epsilon*255,noise_prop,batch_size))

    train_transforms = transforms.Compose([
        transforms.Resize([resize,resize]),
        # transforms.RandomCrop(resize, padding=10),
        # transforms.RandomRotation(10),
        # transforms.GaussianBlur(kernel_size=5, sigma=1.0),# int(sys.argv[2])/2
        # transforms.RandomAdjustSharpness(2,0.5),
        # AddPepperNoise(0.5,0.5),
        # Cutout(2, 112, 1),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        ])
    test_trasforms = transforms.Compose([
        transforms.Resize([resize,resize]),
        transforms.ToTensor(),
        ])
    print('train_transforms:\n',train_transforms)
    print('test_trasforms:\n',test_trasforms)

    root = '/data/zhangzhiling/'+datasetname
    train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',train_transforms)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False)
    test_dataset = torchvision.datasets.ImageFolder(root+'/test_clean',test_trasforms)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) # random
    train_dataset_noise = torchvision.datasets.ImageFolder(root+'/train'+method,train_transforms) # +'_png' str(noise_prop) +str(resize)
    train_dataloader_noise = DataLoader(train_dataset_noise,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) 
    test_dataset_noise = torchvision.datasets.ImageFolder(root+'/test'+method,test_trasforms)
    test_dataloader_noise = DataLoader(test_dataset_noise,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False)
    # if not check_size(train_dataset_noise):
    #     print('size unmatch!')
    #     break
    print('train_dataset:',train_dataset.root)
    print('test_dataset:',test_dataset.root)
    print('train_dataset_noise:',train_dataset_noise.root)
    print('test_dataset_noise:',test_dataset_noise.root)
    num_classes=len(train_dataset.classes)
    # model = torchvision.models.resnet18(num_classes=len(train_dataset.classes)).to(device)
    # if modelname=='mobilenet' or modelname=='inception_resnetv1':
    #     model = Facenet(backbone=modelname, num_classes=num_classes).to(device)  
    #     loss_fun = lambda logits, labels : F.cross_entropy(logits, labels)
        # loss_fun = lambda logits, labels : torch.nn.NLLLoss()(F.log_softmax(logits, dim = -1), labels)  
        # if pretrain_model:
        #     model = model_load(model,modelname)
    if modelname=='mobilenet_v2':
        # inverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 16, 1, 1],
        #     [1, 24, 2, 2],
        #     [1, 32, 3, 2],
        #     [1, 64, 4, 2],
        #     [1, 96, 3, 1],
        #     [1, 160, 3, 2],
        #     [1, 320, 1, 1],
        #     [1, 512, 3, 2],
        # ]
        model = torchvision.models.mobilenet_v2(num_classes=num_classes, block=InvertedResidual).to(device)
        # model = torchvision.models.mobilenet_v2(num_classes=num_classes,inverted_residual_setting=inverted_residual_setting).to(device)
        # model = torchvision.models.mobilenet_v2(num_classes=num_classes).to(device)
    elif modelname=='mobilenet_v1':
        from nets.mobilenet import MobileNetV1
        model = MobileNetV1(num_classes=num_classes).to(device)
    elif modelname=='inception_v3':
        model = torchvision.models.inception_v3(num_classes=num_classes,aux_logits=False,init_weights=True).to(device)
    elif modelname=='resnet18':
        model = torchvision.models.resnet18(num_classes=num_classes).to(device)
        if resize <= 112:
            model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
            model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).to(device)
    elif modelname=='resnet50':
        model = torchvision.models.resnet._resnet('resnet50', Bottleneck, [3, 4, 6, 3], False, True, num_classes=num_classes).to(device)
        # model = torchvision.models.resnet50(num_classes=num_classes).to(device)
    elif modelname=='resnet34':
        model = torchvision.models.resnet34(num_classes=num_classes).to(device)
    
    loss_fun = lambda logits, labels : F.cross_entropy(logits, labels)
    print(model)

    train_itr = len(train_dataloader)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_model, T_max = epochs_test)
    print('modelname:', modelname)
    print('num_classes:', num_classes)
    model_eval(0)
    for epoch in range(1, epochs_test+1):
        loss_sum = 0
        model.train()
        for data in train_dataloader_noise: # _noise
            images, labels = data
            # images, labels_a, labels_b, lam = Cutmix_data(images, labels, 2, 112) # Mixup_data(images, labels,1) # # 
            # images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)
            images, labels = images.to(device), labels.to(device)
            if adv_train:
                images = get_advimg(images,labels)
            logits = model(images)
            loss_model = loss_fun(logits, labels)
            # loss_model = mix_criterion(loss_fun, logits, labels_a, labels_b, lam)
            optimizer_model.zero_grad()
            loss_model.backward()
            optimizer_model.step()
            scheduler.step()
            loss_sum += loss_model
        time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        print(time_now, 'Loss in Epoch {}: {:.4f}'.format(epoch, loss_sum/train_itr))
        if epoch %4 == 1:
            time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            print(time_now, 'lr in Epoch {}:{:.3e}'.format(epoch, optimizer_model.param_groups[0]['lr'])) # 打印学习率
            model_eval(epoch)
        # if epoch%10 == 0:
        #     TNSE_view(train_dataloader,model)
        if loss_sum/train_itr<0.0001:
            model_eval(epoch)
            break
        # if epoch%200==0:# 保存
        #     netG_file_name = './ul_models/mobilenet_arcface_Webface10.pth'
        #     torch.save(model.state_dict(), netG_file_name)
        #     print(netG_file_name,'saved!')
    # save_excel()