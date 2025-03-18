'''
Use Kmeans to get pseudo label
'''

from sklearn.cluster import KMeans
import numpy as np
from models import Facenet
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kmeans_acc(num_classes,label_list):
    '''计算每个类最常出现的预测类别的数目,前提是dataloader顺序读'''
    correct = 0
    num_samples = len(label_list)
    per_samples = int(num_samples/num_classes)
    for i in range(num_classes):
        correct += np.max(np.bincount(label_list[per_samples*i:per_samples*(i+1)]))
    return correct/num_samples

facenet = Facenet(backbone="mobilenet",num_classes=10,embedding_size=128).to(device)
model_dict = facenet.state_dict()
pretrained_path = './nets/facenet_mobilenet.pth' 
pretrained_dict = torch.load(pretrained_path)
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
facenet.load_state_dict(model_dict)
facenet.eval()
datasetname = 'WebFace10' 
root = './'+datasetname
transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    ])
train_dataset = torchvision.datasets.ImageFolder(root+'/train_clean',transform) # train_clean
train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=False,num_workers=1,drop_last=False)
test_dataset =torchvision.datasets.ImageFolder(root+'/test_clean',transform) # test_clean
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=1,drop_last=False)
num_classes = len(train_dataset.classes)
with torch.no_grad():
    logits_list = torch.ones((0,128)).to(device) # 初始化空tensor
    label_list = torch.ones((0)).to(device)
    for _, data in enumerate(train_dataloader, 0):
        image, label = data
        image, label = image.to(device), label.to(device)
        logits_list = torch.cat((logits_list, facenet.feature_extract(image)))   # .feature_extract
        label_list = torch.cat((label_list, label))
    kmeans = KMeans(n_clusters=num_classes,n_init=10).fit(logits_list.cpu())
    train_acc = kmeans_acc(num_classes,kmeans.labels_)
    print('train_acc: {:.3f}'.format(train_acc))
    pre_label_list = torch.ones((0)).to(device) # 初始化空tensor
    label_list = torch.ones((0)).to(device)
    for _, data in enumerate(test_dataloader, 0):
        image, label = data
        image, label = image.to(device), label.to(device)
        pre_label = kmeans.predict(facenet.feature_extract(image).cpu()) #.feature_extract
        pre_label_list = torch.cat((pre_label_list, torch.from_numpy(pre_label).to(device)))
        label_list = torch.cat((label_list, label))
    test_acc = kmeans_acc(num_classes, pre_label_list.cpu())
    print('test_acc: {:.3f}'.format(test_acc))
    name = 'ul_models/'+datasetname+'_k'+str(kmeans.n_clusters)+'.pickle'
    f = open(name,'wb') # https://blog.csdn.net/comli_cn/article/details/107519413
    pickle.dump(kmeans,f)
    print('save '+name+'!')
    f.close()
