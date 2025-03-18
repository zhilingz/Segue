import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mobilenet import MobileNetV1
import pickle

# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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

class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        self.num_classes = num_classes
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
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
    def __init__(self,gen_input_nc,image_nc,num_classes,kmeans_label=True,datasetname=None):
        super(Generator, self).__init__()

        self.num_classes = num_classes
        self.kmeans_label = kmeans_label
        self.embedding_size = 128
        self.bin_num = 16 # bin编码的位数
        ex_channel = self.bin_num 
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
        Csqueeze_lis = [
            nn.Conv2d(32+ex_channel, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        ]
        bottle_neck_lis = [
            ResnetBlock(dim=32),
            ResnetBlock(dim=32),
            ResnetBlock(dim=32),
            ResnetBlock(dim=32)
            ]
        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.Csqueeze = nn.Sequential(*Csqueeze_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        if kmeans_label: # 不需要类别信息
            self.facenet = Facenet(backbone="mobilenet",num_classes=num_classes,embedding_size=self.embedding_size)
            f = open('ul_models/'+datasetname+'_k'+str(num_classes)+'.pickle','rb')
            self.kmeans = pickle.load(f)
            f.close()

    def forward(self, x, labels):
        if self.kmeans_label:
            with torch.no_grad():
                features = self.facenet.feature_extract(x).detach()
                lab = self.kmeans.predict(features.cpu())
            self.lab = torch.from_numpy(lab).cuda().long()
            labels = self.lab
        pred_lab = torch.zeros((labels.shape[0]),self.bin_num).cuda()
        for i, label in enumerate(labels):
            bin_lab = bin(label)
            for j, bit in enumerate(bin_lab[::-1]):
                if bit == 'b':
                    break
                pred_lab[i][-j-1] = int(bit)
        x = self.encoder(x) # x:[-,3,32,32]->[-,32,6,6]或[,3,224,224]->[,32,54,54]
        # [-,10]->[-,10,6,6] 在H和W方向上扩展
        pred_lab = pred_lab.unsqueeze(2).expand(x.shape[0],pred_lab.shape[1],x.shape[2]).unsqueeze(2).expand(
            x.shape[0],pred_lab.shape[1],x.shape[2],x.shape[3])
        x = torch.cat([x,pred_lab],dim=1) # 在C维度上concat，x:[-,32,6,6]->[-,42,6,6]
        x = self.Csqueeze(x) # x:[-,42,6,6]->[-,32,6,6]
        x = self.bottle_neck(x) # 尺寸不变
        x = self.decoder(x)
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