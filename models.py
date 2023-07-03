import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import functools
from nets.inception_resnetv1 import InceptionResnetV1
from nets.mobilenet import MobileNetV1
# import numpy as np
import pickle

# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ArcFace(nn.Module):
    '''https://blog.csdn.net/qq_21539375/article/details/113449055'''
    def __init__(self,feature_dim=1028,cls_dim=10):
        super(ArcFace, self).__init__()
        #x是（N，V）结构，那么W是（V,C结构），V是特征的维度，C是代表类别数
        self.W = nn.Parameter(torch.randn(feature_dim,cls_dim))

    def forward(self,feature,m=0.05,s=10):
        x = F.normalize(feature,dim=1) 
        w = F.normalize(self.W,dim=0)
        cos = torch.clamp(torch.matmul(x,w),-1,1) #（N,C）/10
        a = torch.acos(cos) #(N,C)
        top = torch.exp(s*torch.cos(a+m))  #(N,C)
        #第一项(N,1)  keepdim=True保持形状不变.这是我们原有的softmax的分布。第二项(N,C),最后结果是(N,C)
        down = torch.sum(torch.exp(s*torch.cos(a)),dim=1,keepdim=True)-torch.exp(s*torch.cos(a))
        out = torch.log(top/(top+down))  #(N,C)
        # out = out.sub_(torch.mean(out, dim=1)[:, None]).div_(torch.std(out, dim=1)[:, None])
        return out

class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x

class inception_resnet(nn.Module):
    def __init__(self):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x
        
class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        self.num_classes = num_classes
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet()
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        self.arcnet = ArcFace(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        # x = self.arcnet(x)
        # x = F.normalize(x, p=2, dim=1)
        return x
    
    def feature_extract(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class Generator(nn.Module):
    def __init__(self,gen_input_nc,image_nc,label_feature,num_classes,bin_label=True,kmeans_label=True,datasetname=None):
        super(Generator, self).__init__()

        self.label_feature = label_feature
        self.num_classes = num_classes
        self.bin_label = bin_label
        self.kmeans_label = kmeans_label
        self.embedding_size = 128
        self.bin_num = 16 # bin编码的位数
        ex_channel = self.bin_num if bin_label else num_classes
        norm_layer = nn.BatchNorm2d # BatchNorm2d InstanceNorm2d LayerNorm
        kernelsize1and1 = True # 是否使用1*1卷积
        groupsconv = True # 是否使用分组卷积
        channel_setting_encoder = [
            #ci   co   s
            [3,   8,   1],
            [8,   32,  2],
            [32,  128,  2],
            # [32,  64,  2],
            # [64,  128, 2],
            # [128, 256, 2],
            # [256, 512, 2],
        ]
        channel_setting_decoder = [
            # ci  co   k  s
            # [512, 256, 3, 2],
            # [256, 128, 4, 2],
            # [128,  64, 4, 2],
            # [64,   32, 4, 2],
            # [32,   16, 3,2],# 4, 2],
            # [16,    8, 3,2],# 4, 2],
            # [8,     3, 6,1],# 3, 1],
            [128,   32, 3,2],# 4, 2],
            [32,    8, 3,2],# 4, 2],
            [8,     3, 6,1],# 3, 1],
        ]
        encoder_lis = []
        for setting in channel_setting_encoder:
            encoder_lis += self._make_layer_encoder(norm_layer,setting[0],setting[1],setting[2],kernelsize1and1,groupsconv)
        decoder_lis = []
        for setting in channel_setting_decoder:
            decoder_lis += self._make_layer_decoder(norm_layer,setting[0],setting[1],setting[2],setting[3],kernelsize1and1,groupsconv)
        
        # encoder_lis = [
        #     # 3 224 224
        #     nn.Conv2d(gen_input_nc, 8, kernel_size=1, stride=1, padding=0, bias=True),
        #     # nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=1, bias=True),
        #     norm_layer(8),
        #     nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0, groups=8, bias=True),
        #     norm_layer(8),
        #     nn.ReLU(),
        #     # 8 222 222
        #     nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=True),
        #     # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
        #     norm_layer(16),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0, groups=16, bias=True),
        #     norm_layer(16),
        #     nn.ReLU(),
        #     # 16 110 110
        #     nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=True),
        #     # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #     norm_layer(32),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, groups=32, bias=True),
        #     norm_layer(32),
        #     nn.ReLU(),
        #     # 32 54 54 
        #     nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True),
        #     norm_layer(64),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, groups=64, bias=True),
        #     norm_layer(64),
        #     nn.ReLU(),
        #     # 64 26 26
        #     nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True),
        #     norm_layer(128),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, groups=128, bias=True),
        #     norm_layer(128),
        #     nn.ReLU(),
        #     # 128 12 12
        #     nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=True),
        #     norm_layer(256),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, groups=256, bias=True),
        #     norm_layer(256),
        #     nn.ReLU(),
        #     # 256 5 5
        #     nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=True),
        #     norm_layer(512),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0, groups=512, bias=True),
        #     norm_layer(512),
        #     nn.ReLU(),
        #     # 256 2 2
        # ]
        bottle_channel = channel_setting_encoder[-1][1]
        Csqueeze_lis = [
            nn.Conv2d(bottle_channel+ex_channel, bottle_channel, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(bottle_channel),
            nn.ReLU(),
        ]
        bottle_neck_lis = [
            ResnetBlock(dim=bottle_channel, groupsconv=groupsconv),
            ResnetBlock(dim=bottle_channel, groupsconv=groupsconv),
            ResnetBlock(dim=bottle_channel, groupsconv=groupsconv),
            ResnetBlock(dim=bottle_channel, groupsconv=groupsconv),
            ]
        # decoder_lis = [
        #     # 256 2 2
        #     nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=0, groups=512, bias=False),
        #     norm_layer(512),
        #     nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
        #     norm_layer(256),
        #     nn.ReLU(),
        #     # 256 5 5
        #     nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=0, groups=256, bias=False),
        #     norm_layer(256),
        #     nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
        #     norm_layer(128),
        #     nn.ReLU(),
        #     # 128 12 12
        #     nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=0, groups=128, bias=False),
        #     norm_layer(128),
        #     nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     norm_layer(64),
        #     nn.ReLU(),
        #     # 64 26 26
        #     nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0, groups=64, bias=False),
        #     norm_layer(64),
        #     nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
        #     norm_layer(32),
        #     nn.ReLU(),
        #     # 32 54 54
        #     nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0, groups=32, bias=False),
        #     norm_layer(32),
        #     # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #     norm_layer(16),
        #     nn.ReLU(),
        #     # 16 110 110
        #     nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=0, groups=16, bias=False),
        #     norm_layer(16),
        #     # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1, padding=0, bias=False),
        #     norm_layer(8),
        #     nn.ReLU(),
        #     # 8 222 222
        #     nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1, padding=0, groups=8, bias=False),
        #     norm_layer(8),
        #     # nn.ConvTranspose2d(8, image_nc, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Tanh()
        #     # 3 224 224
        # ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.Csqueeze = nn.Sequential(*Csqueeze_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        if label_feature:
            # if bin_label:
            #     self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
            if kmeans_label: # 不需要类别信息
                self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
                f = open('ul_models/'+datasetname+'_k'+str(num_classes)+'.pickle','rb')
                self.kmeans = pickle.load(f)
                f.close()

    def _make_layer_encoder(self, norm_layer, input_c, output_c, stride, kernelsize1and1, groupsconv) -> nn.Sequential:
        layers = []
        if kernelsize1and1:
            layers.append(
                nn.Conv2d(input_c, output_c, kernel_size=1, stride=1, padding=0, bias=True),
            )
        else:
            layers.append(
                nn.Conv2d(input_c, output_c, kernel_size=3, stride=1, padding=1, bias=True),
            )

        layers.append(
            norm_layer(output_c),
        )
        if groupsconv:
            layers.append(
                nn.Conv2d(output_c, output_c, kernel_size=3, stride=stride, padding=0, groups=output_c, bias=True),
            )    
            
        else:
            layers.append(
                nn.Conv2d(output_c, output_c, kernel_size=3, stride=stride, padding=0, bias=True),
            )
        layers.append(
            norm_layer(output_c),
        )
        layers.append(
            nn.ReLU(),
        )
        return layers

    def _make_layer_decoder(self, norm_layer, input_c, output_c, kernel_size, stride, kernelsize1and1, groupsconv) -> nn.Sequential:
        layers = []
        if groupsconv:
            layers.append(
                nn.ConvTranspose2d(input_c, input_c, kernel_size=kernel_size, stride=stride, padding=0, groups=input_c, bias=False)
            )
        else:
            layers.append(
                nn.ConvTranspose2d(input_c, input_c, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
            )
        layers.append(
            norm_layer(input_c),
        )
        if kernelsize1and1:
            layers.append(
                nn.ConvTranspose2d(input_c, output_c, kernel_size=1, stride=1, padding=0, bias=False),
            )
        else:
            layers.append(
                nn.ConvTranspose2d(input_c, output_c, kernel_size=3, stride=1, padding=1, bias=False),
            )
        layers.append(
            norm_layer(output_c),
        )
        layers.append(
            nn.ReLU(),
        )
        return layers

    def forward(self, x, labels):
        # labels=(labels+3)%10 # 将一个类的噪声加到另一个类上，看噪声是否有效
        if self.label_feature:
            if self.kmeans_label:
                with torch.no_grad():
                    features = self.facenet.feature_extract(x).detach()
                    lab = self.kmeans.predict(features.cpu())
                self.lab = torch.from_numpy(lab).cuda()
                labels = self.lab
                # pred_lab = F.one_hot(self.lab.to(torch.int64), self.num_classes)
            if self.bin_label:
                pred_lab = torch.zeros((labels.shape[0]),self.bin_num).cuda()
                for i, label in enumerate(labels):
                    bin_lab = bin(label)
                    # for j, bit in enumerate(bin_lab[2:]): # 0b10和0b1是同一个pred_lab
                    #     pred_lab[i][-j] = int(bit)
                    for j, bit in enumerate(bin_lab[::-1]):
                        if bit == 'b':
                            break
                        pred_lab[i][-j-1] = int(bit)
            else:
                # labels:[-]->[-,10] 变成onehot向量
                pred_lab = F.one_hot(labels, self.num_classes)
            x = self.encoder(x) # x:[-,3,32,32]->[-,32,6,6]或[,3,224,224]->[,32,54,54]
            # [-,10]->[-,10,6,6] 在H和W方向上扩展
            pred_lab = pred_lab.unsqueeze(2).expand(x.shape[0],pred_lab.shape[1],x.shape[2]).unsqueeze(2).expand(
                x.shape[0],pred_lab.shape[1],x.shape[2],x.shape[3])
            x = torch.cat([x,pred_lab],dim=1) # 在C维度上concat，x:[-,32,6,6]->[-,42,6,6]
            x = self.Csqueeze(x) # x:[-,42,6,6]->[-,32,6,6]
            x = self.bottle_neck(x) # 尺寸不变
            x = self.decoder(x)
        else:
            x = self.encoder(x)
            x = self.bottle_neck(x) # 尺寸不变
            x = self.decoder(x)
        return x

# class Generator(nn.Module):
#     def __init__(self,gen_input_nc,image_nc,label_feature,num_classes,bin_label=True,kmeans_label=True,datasetname=None):
#         super(Generator, self).__init__()

#         self.label_feature = label_feature
#         self.num_classes = num_classes
#         self.bin_label = bin_label
#         self.kmeans_label = kmeans_label
#         self.embedding_size = 128
#         self.bin_num = 16 # bin编码的位数
#         ex_channel = self.bin_num if bin_label else num_classes
#         encoder_lis = [
#             # 3 224 224
#             nn.Conv2d(gen_input_nc, 8, kernel_size=1, stride=1, padding=0, bias=True),
#             # nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.InstanceNorm2d(8),
#             nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0, groups=8, bias=True),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # 8 222 222
#             nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=True),
#             # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.InstanceNorm2d(16),
#             nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0, groups=16, bias=True),
#             nn.InstanceNorm2d(16),
#             nn.ReLU(),
#             # 16 110 110
#             nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=True),
#             # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.InstanceNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, groups=32, bias=True),
#             nn.InstanceNorm2d(32),
#             nn.ReLU(),
#             # 32 54 54 
#             nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, groups=64, bias=True),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(),
#             # 64 26 26
#             nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(128),
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, groups=128, bias=True),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(),
#             # 128 12 12
#             nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(256),
#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, groups=256, bias=True),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(),
#             # 256 5 5
#             nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(512),
#             nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0, groups=512, bias=True),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(),
#         ]
#         bottle_channel = 512
#         Csqueeze_lis = [
#             nn.Conv2d(bottle_channel+ex_channel, bottle_channel, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.InstanceNorm2d(bottle_channel),
#             nn.ReLU(),
#         ]
#         bottle_neck_lis = [
#             ResnetBlock(bottle_channel),
#             ResnetBlock(bottle_channel),
#             ResnetBlock(bottle_channel),
#             ResnetBlock(bottle_channel),
#             ]
#         decoder_lis = [
#             nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=0, groups=512, bias=False),
#             nn.InstanceNorm2d(512),
#             nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(),
#             # 256 5 5
#             nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=0, groups=256, bias=False),
#             nn.InstanceNorm2d(256),
#             nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(),
#             # 128 12 12
#             nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=0, groups=128, bias=False),
#             nn.InstanceNorm2d(128),
#             nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(),
#             # 64 26 26
#             nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0, groups=64, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(32),
#             nn.ReLU(),
#             # 32 54 54
#             nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0, groups=32, bias=False),
#             nn.InstanceNorm2d(32),
#             # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(16),
#             nn.ReLU(),
#             # 16 110 110
#             nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=0, groups=16, bias=False),
#             nn.InstanceNorm2d(16),
#             # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # 8 222 222
#             nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1, padding=0, groups=8, bias=False),
#             nn.InstanceNorm2d(8),
#             # nn.ConvTranspose2d(8, image_nc, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ConvTranspose2d(8, image_nc, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Tanh()
#             # 3 224 224
#         ]

#         self.encoder = nn.Sequential(*encoder_lis)
#         self.Csqueeze = nn.Sequential(*Csqueeze_lis)
#         self.bottle_neck = nn.Sequential(*bottle_neck_lis)
#         self.decoder = nn.Sequential(*decoder_lis)
#         if label_feature:
#             # if bin_label:
#             #     self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
#             if kmeans_label: # 不需要类别信息
#                 self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
#                 f = open('ul_models/'+datasetname+'_k'+str(num_classes)+'.pickle','rb')
#                 self.kmeans = pickle.load(f)
#                 f.close()

#     def forward(self, x, labels):
#         # labels=(labels+3)%10 # 将一个类的噪声加到另一个类上，看噪声是否有效
#         if self.label_feature:
#             if self.kmeans_label:
#                 with torch.no_grad():
#                     features = self.facenet.feature_extract(x).detach()
#                     lab = self.kmeans.predict(features.cpu())
#                 self.lab = torch.from_numpy(lab).cuda()
#                 labels = self.lab
#                 # pred_lab = F.one_hot(self.lab.to(torch.int64), self.num_classes)
#             if self.bin_label:
#                 pred_lab = torch.zeros((labels.shape[0]),self.bin_num).cuda()
#                 for i, label in enumerate(labels):
#                     bin_lab = bin(label)
#                     # for j, bit in enumerate(bin_lab[2:]): # 0b10和0b1是同一个pred_lab
#                     #     pred_lab[i][-j] = int(bit)
#                     for j, bit in enumerate(bin_lab[::-1]):
#                         if bit == 'b':
#                             break
#                         pred_lab[i][-j-1] = int(bit)
#             else:
#                 # labels:[-]->[-,10] 变成onehot向量
#                 pred_lab = F.one_hot(labels, self.num_classes)
#             x = self.encoder(x) # x:[-,3,32,32]->[-,32,6,6]或[,3,224,224]->[,32,54,54]
#             # [-,10]->[-,10,6,6] 在H和W方向上扩展
#             pred_lab = pred_lab.unsqueeze(2).expand(x.shape[0],pred_lab.shape[1],x.shape[2]).unsqueeze(2).expand(
#                 x.shape[0],pred_lab.shape[1],x.shape[2],x.shape[3])
#             x = torch.cat([x,pred_lab],dim=1) # 在C维度上concat，x:[-,32,6,6]->[-,42,6,6]
#             x = self.Csqueeze(x) # x:[-,42,6,6]->[-,32,6,6]
#             x = self.bottle_neck(x) # 尺寸不变
#             x = self.decoder(x)
#         else:
#             x = self.encoder(x)
#             x = self.bottle_neck(x) # 尺寸不变
#             x = self.decoder(x)
#         return x

# class Generator(nn.Module):
#     def __init__(self,gen_input_nc,image_nc,label_feature,num_classes,bin_label=True,kmeans_label=True,datasetname=None):
#         super(Generator, self).__init__()

#         self.label_feature = label_feature
#         self.num_classes = num_classes
#         self.bin_label = bin_label
#         self.kmeans_label = kmeans_label
#         self.embedding_size = 128
#         self.bin_num = 16 # bin编码的位数
#         ex_channel = self.bin_num if bin_label else num_classes
        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]
#         # encoder_lis = [
#         #     # MNIST:1*28*28
#         #     nn.Conv2d(gen_input_nc, 8, kernel_size=1, stride=1, padding=0, bias=True),
#         #     # nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=1, bias=True),
#         #     nn.InstanceNorm2d(8),
#         #     nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0, groups=8, bias=True),
#         #     nn.InstanceNorm2d(8),
#         #     nn.ReLU(),
#         #     # 8*26*26
#         #     nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=True),
#         #     # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
#         #     nn.InstanceNorm2d(16),
#         #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0, groups=16, bias=True),
#         #     nn.InstanceNorm2d(16),
#         #     nn.ReLU(),
#         #     # 16*12*12
#         #     nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=True),
#         #     # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
#         #     nn.InstanceNorm2d(32),
#         #     nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, groups=32, bias=True),
#         #     nn.InstanceNorm2d(32),
#         #     nn.ReLU(),
#         #     # 32*5*5
#         # ]
#         Csqueeze_lis = [
#             nn.Conv2d(32+ex_channel, 32, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.InstanceNorm2d(32),
#             nn.ReLU(),
#         ]
#         bottle_neck_lis = [
#             ResnetBlock(32),
#             ResnetBlock(32),
#             ResnetBlock(32),
#             ResnetBlock(32),
#             ]
#         # decoder_lis = [
#         #     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
#         #     nn.InstanceNorm2d(16),
#         #     nn.ReLU(),
#         #     # state size. 16 x 11 x 11
#         #     nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
#         #     nn.InstanceNorm2d(8),
#         #     nn.ReLU(),
#         #     # state size. 8 x 23 x 23
#         #     nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
#         #     nn.Tanh()
#         #     # state size. image_nc x 28 x 28
#         # ]
#         decoder_lis = [
#             nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=0, groups=32, bias=False),
#             nn.InstanceNorm2d(32),
#             # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(16),
#             nn.ReLU(),
#             # state size. 16 x 11 x 11
#             nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0, groups=16, bias=False),
#             nn.InstanceNorm2d(16),
#             # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # state size. 8 x 23 x 23
#             nn.ConvTranspose2d(8, 8, kernel_size=6, stride=1, padding=0, groups=8, bias=False),
#             nn.InstanceNorm2d(8),
#             # nn.ConvTranspose2d(8, image_nc, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ConvTranspose2d(8, image_nc, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Tanh()
#             # state size. image_nc x 28 x 28
#         ]

#         self.encoder = nn.Sequential(*encoder_lis)
#         self.Csqueeze = nn.Sequential(*Csqueeze_lis)
#         self.bottle_neck = nn.Sequential(*bottle_neck_lis)
#         self.decoder = nn.Sequential(*decoder_lis)
#         if label_feature:
#             # if bin_label:
#             #     self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
#             if kmeans_label: # 不需要类别信息
#                 self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
#                 f = open('ul_models/'+datasetname+'_k'+str(num_classes)+'.pickle','rb')
#                 self.kmeans = pickle.load(f)
#                 f.close()

#     def forward(self, x, labels):
#         # labels=(labels+3)%10 # 将一个类的噪声加到另一个类上，看噪声是否有效
#         if self.label_feature:
#             if self.kmeans_label:
#                 with torch.no_grad():
#                     features = self.facenet.feature_extract(x).detach()
#                     lab = self.kmeans.predict(features.cpu())
#                 self.lab = torch.from_numpy(lab).cuda()
#                 labels = self.lab
#                 # pred_lab = F.one_hot(self.lab.to(torch.int64), self.num_classes)
#             if self.bin_label:
#                 pred_lab = torch.zeros((labels.shape[0]),self.bin_num).cuda()
#                 for i, label in enumerate(labels):
#                     bin_lab = bin(label)
#                     # for j, bit in enumerate(bin_lab[2:]): # 0b10和0b1是同一个pred_lab
#                     #     pred_lab[i][-j] = int(bit)
#                     for j, bit in enumerate(bin_lab[::-1]):
#                         if bit == 'b':
#                             break
#                         pred_lab[i][-j-1] = int(bit)
#             else:
#                 # labels:[-]->[-,10] 变成onehot向量
#                 pred_lab = F.one_hot(labels, self.num_classes)
#             x = self.encoder(x) # x:[-,3,32,32]->[-,32,6,6]或[,3,224,224]->[,32,54,54]
#             # [-,10]->[-,10,6,6] 在H和W方向上扩展
#             pred_lab = pred_lab.unsqueeze(2).expand(x.shape[0],pred_lab.shape[1],x.shape[2]).unsqueeze(2).expand(
#                 x.shape[0],pred_lab.shape[1],x.shape[2],x.shape[3])
#             x = torch.cat([x,pred_lab],dim=1) # 在C维度上concat，x:[-,32,6,6]->[-,42,6,6]
#             x = self.Csqueeze(x) # x:[-,42,6,6]->[-,32,6,6]
#             x = self.bottle_neck(x) # 尺寸不变
#             x = self.decoder(x)
#         else:
#             x = self.encoder(x)
#             x = self.bottle_neck(x) # 尺寸不变
#             x = self.decoder(x)
#         return x

class SimSiamModel(nn.Module):
    def __init__(self):
        from nets.resnet_cifar import model_dict  
        super().__init__()
        
        model_fun, feat_dim = model_dict['resnet18']
        self.backbone = model_fun()
        self.feat_dim = feat_dim
        self.proj_head = nn.Sequential(
                nn.Linear(self.feat_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),

                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
            )
        self.pred_head = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 2048),
                )
        
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.ToTensor(),
        ])
    
    def negcos(self, p1, p2, z1, z2):
        p1 = F.normalize(p1, dim=1); p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
        return - 0.5 * ((p1*z2.detach()).sum(dim=1).mean() + (p2*z1.detach()).sum(dim=1).mean())

    def k_grad_negcos(self, p1, p2, z1, z2):
        p1 = F.normalize(p1, dim=1); p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
        return - 0.5 * ((p1*z2).sum(dim=1).mean() + (p2*z1).sum(dim=1).mean())

    def forward(self, x):
        x1, x2 = self.transform(x), self.transform(x)
        x1 = self.backbone(x1) # ResNet18, 输出512维向量
        x2 = self.backbone(x2) # ResNet18, 输出512维向量
        z1 = self.proj_head(x1)
        z2 = self.proj_head(x2)
        p1 = self.pred_head(z1)
        p2 = self.pred_head(z2)
        return self.negcos(p1, p2, z1, z2)
    
    def k_grad_forward(self, x):
        x1, x2 = self.transform(x), self.transform(x)
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        z1 = self.proj_head(x1)
        z2 = self.proj_head(x2)
        p1 = self.pred_head(z1)
        p2 = self.pred_head(z2)
        return self.k_grad_negcos(p1, p2, z1, z2)

# class Generator(nn.Module):
#     def __init__(self,gen_input_nc,image_nc,label_feature,num_classes,bin_label=True,kmeans_label=True,datasetname=None):
#         super(Generator, self).__init__()

#         self.label_feature = label_feature
#         self.num_classes = num_classes
#         self.bin_label = bin_label
#         self.kmeans_label = kmeans_label
#         self.embedding_size = 128
        
#         encoder_lis = [
#             # 224
#             nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # 222
#             nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=0, bias=True),
#             nn.InstanceNorm2d(32),
#             nn.ReLU(),
#             # 110
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0, bias=True), 
#             nn.InstanceNorm2d(64),
#             nn.ReLU(),
#             # 54
#         ]
#         Csqueeze_lis = [
#             nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(),
#         ]
#         bottle_neck_lis = [
#             ResnetBlock(64),
#             ResnetBlock(64),
#             ResnetBlock(64),
#             ResnetBlock(64),
#             ]
#         decoder_lis = [
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.InstanceNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 8, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
#             nn.Tanh()
#         ]

#         self.encoder = nn.Sequential(*encoder_lis)
#         self.Csqueeze = nn.Sequential(*Csqueeze_lis)
#         self.bottle_neck = nn.Sequential(*bottle_neck_lis)
#         self.decoder = nn.Sequential(*decoder_lis)
#         if self.bin_label:
#             self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
#             # # model_path = 'ul_models/facenet_Webface10.pth'
#             # model_path = 'pre_models/facenet_mobilenet.pth'
#             # model_dict = self.facenet.state_dict()
#             # pretrained_dict = torch.load(model_path)
#             # load_key, no_load_key, temp_dict = [], [], {}
#             # for k, v in pretrained_dict.items():
#             #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#             #         temp_dict[k] = v
#             #         load_key.append(k)
#             #     else:
#             #         no_load_key.append(k)
#             # model_dict.update(temp_dict)
#             # self.facenet.load_state_dict(model_dict)
#             # for p in self.facenet.parameters(): # 将需要冻结的参数的 requires_grad 设置为 False
#             #     p.requires_grad = False
#             # self.facenet.eval()
#         if self.kmeans_label: # 不需要类别信息
#             self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
#             # model_path = 'pre_models/facenet_mobilenet.pth' # ul_models/mobilenet_arcface_Webface10.pth pre_models/facenet_mobilenet.pth
#             # model_dict = self.facenet.state_dict()
#             # pretrained_dict = torch.load(model_path)
#             # load_key, no_load_key, temp_dict = [], [], {}
#             # for k, v in pretrained_dict.items():
#             #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#             #         temp_dict[k] = v
#             #         load_key.append(k)
#             #     else:
#             #         no_load_key.append(k)
#             # model_dict.update(temp_dict)
#             # self.facenet.load_state_dict(model_dict)
#             # for p in self.facenet.parameters(): # 将需要冻结的参数的 requires_grad 设置为 False
#             #     p.requires_grad = False
#             # self.facenet.eval()
#             # f = open('ul_models/'+datasetname+'_kmeans.pickle','rb')
#             # self.kmeans = pickle.load(f)
#             # f.close()

#     def forward(self, x, labels):
#         # labels=(labels+3)%10 # 将一个类的噪声加到另一个类上，看噪声是否有效
#         if self.label_feature:
#             if self.bin_label:
#                 # with torch.no_grad():
#                 #     lab = self.facenet(x).detach() # [-,10]
#                 # self.lab = torch.argmax(lab,1)
#                 # pred_lab = F.one_hot(self.lab, self.num_classes)
#                 with torch.no_grad():
#                     features = self.facenet.feature_extract(x).detach()
#                 # pred_lab = F.one_hot(features, features.shape[1])
#                 min_vals, _ = torch.min(features, dim=1, keepdim=True)
#                 max_vals, _ = torch.max(features, dim=1, keepdim=True)
#                 # 最小-最大缩放，将x的范围缩放到[0, 1]
#                 features = (features - min_vals) / (max_vals - min_vals)
#                 features *= 10
#                 x = self.encoder(x)
#                 features = features.unsqueeze(2).expand(x.shape[0],features.shape[1],x.shape[2]).unsqueeze(2).expand(
#                     x.shape[0],features.shape[1],x.shape[2],x.shape[3])
#                 x = torch.cat([x,features],dim=1) # x:[-,128,6,6]->[-,256,6,6]
#                 x = self.Csqueeze(x) # x:[-,256,6,6]->[-,128,6,6]
#                 x = self.bottle_neck(x) # 尺寸不变
#                 x = self.decoder(x)
#                 return x
#             elif self.kmeans_label:
#                 with torch.no_grad():
#                     features = self.facenet.feature_extract(x).detach()
#                     lab = self.kmeans.predict(features.cpu())
#                 self.lab = torch.from_numpy(lab).cuda()
#                 pred_lab = F.one_hot(self.lab.to(torch.int64), self.num_classes)
#             else:
#                 # labels:[-]->[-,10] 先变成onehot向量
#                 pred_lab = F.one_hot(labels, self.num_classes)
#             x = self.encoder(x) # x:[-,3,32,32]->[-,32,6,6]或[,3,224,224]->[,32,54,54]
#             # [-,10]->[-,10,6,6] 在H和W方向上扩展
#             pred_lab = pred_lab.unsqueeze(2).expand(x.shape[0],self.num_classes,x.shape[2]).unsqueeze(2).expand(
#                 x.shape[0],self.num_classes,x.shape[2],x.shape[3])
#             x = torch.cat([x,pred_lab],dim=1) # 在C维度上concat，x:[-,32,6,6]->[-,42,6,6]
#             x = self.Csqueeze(x) # x:[-,42,6,6]->[-,32,6,6]
#         x = self.bottle_neck(x) # 尺寸不变
#         x = self.decoder(x)
#         return x

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
            Parameters:
                input_nc (int)      -- the number of channels in input images
                output_nc (int)     -- the number of channels in output images
                ngf (int)           -- the number of filters in the last conv layer
                norm_layer          -- normalization layer
                use_dropout (bool)  -- if use dropout layers
                n_blocks (int)      -- the number of ResNet blocks
                padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero"""
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class FocalLoss(nn.Module):
    '''loss'''
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        logp = torch.nn.CrossEntropyLoss(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class CosFace(nn.Module):
    '''metric'''
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s
    
from typing import Callable, Optional
from torchvision.models.resnet import conv1x1, conv3x3
from torch import Tensor
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups # inplanes
        groups = width
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, width, 1, 1, 1) #
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # self.conv3 = conv3x3(width, planes * self.expansion, 1, 1, 1) #
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
from typing import List
from torchvision.ops.misc import ConvNormActivation
class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # BatchNorm2d InstanceNorm2d LayerNorm

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
            # layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=3, padding=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
            # ConvNormActivation(hidden_dim, hidden_dim, stride=stride, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            # nn.Conv2d(hidden_dim, oup, 3, 1, 1, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
# Target Model definition
class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False, groupsconv=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, groupsconv)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, groupsconv):
        conv_block = []
        p = 0 # 先padding再卷积，不改变图片尺寸
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if groupsconv:
            groups = dim
        else:
            groups = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, groups=groups, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, groups=groups, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out