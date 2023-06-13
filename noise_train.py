import os
import sys
import time
import torch
import torchvision
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0) # 固定随机种子，方便做对照试验
torch.cuda.manual_seed(0)

from dataset import getdataloader
from ul_method import ul_method
from models import Facenet, weights_init, Bottleneck, InvertedResidual

# start_time = time.time()

if sys.argv[3] == 'log':
    print_log = open("logs/"+sys.argv[4],"a",buffering=1)
    sys.stdout = print_log
    print("\n\n", sys.argv)
# train setting
epochs = 40
save_epoch = 10 # 多少epoch保存一次
Epsilon = 8/255 #8/255
noise_prop = 1 # 0到1，毒化率``
label_feature = True # 将类别信息加到图片特征中
bin_label = True # 使用预训练的人脸提取器和arcface得到预测标签，代替真实标签嵌入图片特征中
kmeans_label = False # 使用预训练的人脸提取器提取人脸128维特征，用kmeans聚类预测标签代替真实标签
# dataset setting
root = '/data/zhangzhiling/'
method = sys.argv[1] # GUE UEc UEs TUE RUE
datasetname = "WebFace10"  # sys.argv[2] WebFace50 WebFace10_ ImageNet10 CIFAR10 CIFAR10_0.2 CelebA10 VGGFace10
modelname =  sys.argv[2] #'resnet18' # resnet18 mobilenet_v2 inception_v3
img_size = 32 if 'CIFAR10' in datasetname else 224
batch_size = 128 if method == 'TUE' or 'CIFAR10' in datasetname else 16
print('batch_size',batch_size)
train_dataloader, test_dataloader = getdataloader(root, datasetname, img_size, batch_size)
# unlearnable method
num_classes = len(train_dataloader.dataset.classes)
ul = ul_method(method, train_dataloader, test_dataloader, datasetname, num_classes, img_size, 
               Epsilon, label_feature, bin_label, kmeans_label,0, device) #
# agent model setting
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
    # ]
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
    # (affine=True, track_running_stats=True)
    model = torchvision.models.mobilenet_v2(num_classes=num_classes, block=InvertedResidual).to(device)
    # model = torchvision.models.mobilenet_v2(num_classes=num_classes,inverted_residual_setting=inverted_residual_setting).to(device)
    # model = torchvision.models.mobilenet_v2(num_classes=num_classes).to(device)
    # model = Facenet(backbone=modelname, num_classes=num_classes).to(device)  
    # loss_fun = lambda logits, labels : torch.nn.NLLLoss()(F.log_softmax(logits, dim = -1), labels)  
    # if pretrain_model:
    #     model = model_load(model,modelname)
elif modelname=='mobilenet_v1':
    from nets.mobilenet import MobileNetV1
    model = MobileNetV1(num_classes=num_classes).to(device)
elif modelname=='inception_v3':
    model = torchvision.models.inception_v3(num_classes=num_classes, aux_logits=False,init_weights=True).to(device)
elif modelname=='resnet18':
    model = torchvision.models.resnet18(num_classes=num_classes).to(device)
    if img_size <= 112:
        model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).to(device)
elif modelname=='resnet50':
    model = torchvision.models.resnet._resnet('resnet50', Bottleneck, [3, 4, 6, 3], False, True, num_classes=num_classes).to(device)
    # model = torchvision.models.resnet50(num_classes=num_classes).to(device)
elif modelname=='resnet34':
    model = torchvision.models.resnet34(num_classes=num_classes).to(device)
elif modelname=='resnet152':
    model = torchvision.models.resnet._resnet('resnet152', torchvision.models.resnet.BasicBlock, [3, 8, 36, 3], False, True, num_classes=num_classes).to(device)
    # model = torchvision.models.resnet152(num_classes=num_classes).to(device)
elif modelname=='densenet121':
    model = torchvision.models.densenet121(num_classes=num_classes).to(device)
elif modelname=='vgg11':
    model = torchvision.models.vgg11(num_classes=num_classes).to(device)
elif modelname=='efficientnet_b0':
    model = torchvision.models.efficientnet_b0(num_classes=num_classes).to(device)
    
print(model)
loss_fun = lambda logits, labels : F.cross_entropy(logits, labels)
# model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) #5e-4
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max =  epochs)
print('surrogate model:', modelname)
ul.model_eval(model, 0)
pretrain_epoch = 3
for epoch in range(1, epochs+1):
    ul.loss_init()
    if epoch%5 == 1 or epoch < pretrain_epoch: # 隔几个epoch更新一次目标模型
        model.train()
        ul.idx = 0
        for iter, data in enumerate(train_dataloader):
            image, label = data
            image, label = image.to(device), label.to(device)
            adv_image = ul.add_noise(image, label)
            if method == 'TUE':
                loss_tue = ul.SimSiam_net(adv_image)
                ul.optimizer.zero_grad()
                loss_tue.backward()
                ul.optimizer.step()
            logits = model(adv_image)
            loss_model = F.cross_entropy(logits, label)
            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()
            scheduler.step()
            if iter*batch_size>10000 and datasetname == 'CIFAR10':
                break
        ul.model_eval(model, epoch)
        time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        print(time_now, "epoch %d: lr_model:%.3e"%(epoch, optimizer.param_groups[0]['lr']))
    
    if epoch >= pretrain_epoch:
        ul.idx = 0
        for _, data in enumerate(train_dataloader):
            image, label = data
            ul.train_noise(model, image.to(device), label.to(device))
        ul.print_loss(epoch) # print statistics
        if torch.abs(ul.loss_ul_sum/ul.num_itr) < 0.001:
            ul.model_eval(model, epoch)
            ul.save_noise()
            break
    
    if epoch%save_epoch == 0: # 保存
        ul.save_noise()

# end_time= time.time()

# print('time cost : %.5f sec' %(start_time-end_time))