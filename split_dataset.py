import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    ])
bs = 10
split_rate = 12 # 划分比例
root = '/data/zhangzhiling/WebFace10_/'
output_train = root+'train_clean/'
output_test = root+'test_clean/'
if not os.path.exists(output_train):
    os.mkdir(output_train)
    for i in range(10):
        os.mkdir(output_train+str(i))
if not os.path.exists(output_test):
    os.mkdir(output_test)
    for i in range(10):
        os.mkdir(output_test+str(i))
raw_dataset = torchvision.datasets.ImageFolder(root+'raw',transform)
raw_dataloader = DataLoader(raw_dataset,batch_size=bs,shuffle=False,num_workers=4,drop_last=False)
num_samlpes = len(raw_dataset.targets) # 总样本数
num_classes = len(raw_dataset.classes) # 类别数
iter_per = num_samlpes/num_classes/bs  # 每类iteration数
iter_per = int(iter_per)
id_arr = [0 for _ in range(10)]
for iter, data in enumerate(raw_dataloader):
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    iter %= iter_per
    if iter==split_rate:  # 测试集
        for i, label in enumerate(labels):
            torchvision.utils.save_image(images[i],output_test+str(int(label))+'/'+str(id_arr[label])+'.jpg') 
            id_arr[label]+=1
    else:           # 训练集
        for i, label in enumerate(labels):
            torchvision.utils.save_image(images[i],output_train+str(int(label))+'/'+str(id_arr[label])+'.jpg') 
            id_arr[label]+=1