import os
import shutil
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# root = '/data/zhangzhiling/WebFace/'
# exist_list = []
# exist_list += os.listdir('/data/zhangzhiling/WebFace10/test_clean')
# exist_list += os.listdir('/data/zhangzhiling/WebFace10_/raw')
# dir_list = os.listdir(root)
# num = 0
# for dir in dir_list:
#     len_dir = len(os.listdir(root+dir))
#     # img_list = sorted(os.listdir(root+dir))
#     # idx = 0
#     # for img in img_list:
#     #     img_idx = int(img_list[0].split('.')[0])
#     if len_dir>=150 and dir not in exist_list:
#         shutil.copytree(root+dir, '/data/zhangzhiling/WebFace50/raw/'+dir)
#         print(dir, len_dir)
#         num += 1
#         if num >=50 :
#             break
# print('end')

transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    ]) # webface 图片原尺寸为96，112
bs = 1
root = '/data/zhangzhiling/VGGFace10/'
output_train = root+'train_clean/'
output_test = root+'test_clean/'
raw_dataset = torchvision.datasets.ImageFolder(root+'raw',transform)
raw_dataloader = DataLoader(raw_dataset,batch_size=bs,shuffle=False,num_workers=4,drop_last=False)
num_classes = len(raw_dataset.classes)
rjust = lambda x:str(x).rjust(len(str(num_classes-1)),'0') # input:int, output:str
if not os.path.exists(output_train):
    os.mkdir(output_train)
    for i in range(num_classes):
        os.mkdir(output_train+rjust(i))
if not os.path.exists(output_test):
    os.mkdir(output_test)
    for i in range(num_classes):
        os.mkdir(output_test+rjust(i))
id_arr = [0 for _ in range(num_classes)]
for images, labels in tqdm(raw_dataloader):
    for i, label in enumerate(labels):
        if id_arr[label] >= 150:
            continue
        elif id_arr[label] < 130:  # 训练集
            torchvision.utils.save_image(images[i],output_train+rjust(int(label))+'/'+str(id_arr[label])+'.jpg') 
            id_arr[label]+=1
        else:           # 测试集
            torchvision.utils.save_image(images[i],output_test+rjust(int(label))+'/'+str(id_arr[label])+'.jpg') 
            id_arr[label]+=1