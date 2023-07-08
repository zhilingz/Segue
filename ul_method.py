import os
import sys
import time
import collections
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import Generator
from models import weights_init
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class ul_method():
    def __init__(self,logpath,method,train_dataloader,test_dataloader,datasetname,modelname,num_classes,size,Epsilon,label_feature,bin_label,kmeans_label,rho,device):
        self.logpath = logpath
        self.method = method
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.datasetname = datasetname
        self.modelname = modelname
        self.num_classes = num_classes
        self.Epsilon = Epsilon
        self.size = size
        self.device = device
        self.idx = 0
        if self.method == 'GUE':
            self.lr_noise = 5e-4
            self.atk_Epsilon = int(rho)/255 # 对抗训练噪声的强度应该小于不可学习噪声强度
            self.atk_step_size = self.atk_Epsilon/5 # 对抗训练更新大小
            self.atk_step = 5 
            self.netG = Generator(3,3,label_feature,num_classes,bin_label,kmeans_label,datasetname).to(device)
            # self.netG = UNet(n_channels=3, n_classes=num_classes, kmeans_label=kmeans_label).to(device) #, bilinear=args.bilinear)
            # sum_p = 0
            # for p in self.netG.parameters():
            #     sum_p +=p.numel()
            # initialize all weights
            # self.model_init(label_feature,kmeans_label)
            # self.netG.load_state_dict(torch.load('ul_models/WebFace10_GUE224t10.pth'))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=self.lr_noise)
            self.trans = transforms.Compose([
                    # transforms.GaussianBlur(kernel_size=(5,5), sigma=0.5),
                    # # transforms.RandomAdjustSharpness(5,1),
                    # transforms.RandomRotation(5),
                    # transforms.RandomCrop(224, padding=10),
                    # transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomVerticalFlip(0.5)
                    ])
            print('bin_label:{}, kmeans_label:{}'.format(bin_label, kmeans_label))
            print('distortion layer')
            print(self.trans)
            print(self.netG)
        elif method=='UEc': # 随机初始化10个类的噪声，perturbation的尺寸为[cls,3,size,size]
            self.lr_noise = Epsilon/10
            self.noise = torch.FloatTensor(*[num_classes,3,size,size]).uniform_(-Epsilon,Epsilon).to(device)
        elif method=='UE': # sample-wise
            self.lr_noise = Epsilon/10
            self.num_samples = len(train_dataloader.dataset)
            self.noise = torch.FloatTensor(*[self.num_samples,3,size,size]).uniform_(-Epsilon,Epsilon).to(device)
            # self.noise = torch.load('./ul_models/CIFAR10_0.2UE.pt')
        elif method=='TUE': # sample-wise
            from models import SimSiamModel
            self.SimSiam_net = SimSiamModel().to(device)
            self.optimizer = torch.optim.SGD(self.SimSiam_net.parameters(), lr=0.06, momentum=0.9, weight_decay=5e-4)
            # SimSiam_net.eval()
            self.lr_noise = Epsilon/10
            self.num_samples = len(train_dataloader.dataset)
            self.noise = torch.FloatTensor(*[self.num_samples,3,size,size]).uniform_(-Epsilon,Epsilon).to(device)
            # self.noise = torch.load('ul_models/WebFace10_TUE224.pt')
        elif method=='RUE':
            self.lr_noise = Epsilon/10
            self.atk_Epsilon = int(rho)/255 # 对抗训练噪声的强度应该小于不可学习噪声强度
            self.atk_step_size = self.atk_Epsilon/5 # 对抗训练更新大小
            self.atk_step = 5 # 每一步更新RUE噪声时，对抗训练的次数
            self.num_samples = len(train_dataloader.dataset)
            self.noise = torch.FloatTensor(*[self.num_samples,3,size,size]).uniform_(-Epsilon,Epsilon).to(device)
            print('atk_Epsilon:{}'.format(self.atk_Epsilon*255))
        
        print('method: ',self.method)
        print('lr_noise: {:.4e}'.format(self.lr_noise))

    def model_init(self,label_feature, kmeans_label):
        self.netG.apply(weights_init)
        if label_feature:
                # # model_path = 'ul_models/facenet_Webface10.pth'
                # model_path = 'pre_models/facenet_mobilenet.pth'
                # model_dict = self.netG.facenet.state_dict()
                # pretrained_dict = torch.load(model_path)
                # load_key, no_load_key, temp_dict = [], [], {}
                # for k, v in pretrained_dict.items():
                #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                #         temp_dict[k] = v
                #         load_key.append(k)
                #     else:
                #         no_load_key.append(k)
                # model_dict.update(temp_dict)
                # self.netG.facenet.load_state_dict(model_dict)
                # for p in self.netG.facenet.parameters(): # 将需要冻结的参数的 requires_grad 设置为 False
                #     p.requires_grad = False
                # self.netG.facenet.eval()
            if kmeans_label: # 不需要类别信息
                model_path = 'pre_models/facenet_mobilenet.pth' # ul_models/mobilenet_arcface_Webface10.pth pre_models/facenet_mobilenet.pth
                model_dict = self.netG.facenet.state_dict()
                pretrained_dict = torch.load(model_path)
                load_key, no_load_key, temp_dict = [], [], {}
                for k, v in pretrained_dict.items():
                    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                        temp_dict[k] = v
                        load_key.append(k)
                    else:
                        no_load_key.append(k)
                model_dict.update(temp_dict)
                self.netG.facenet.load_state_dict(model_dict)
                for p in self.netG.facenet.parameters(): # 将需要冻结的参数的 requires_grad 设置为 False
                    p.requires_grad = False
                self.netG.facenet.eval()

    def add_noise(self, image, label):
        '''在这一批图像上添加不可学习噪声'''
        if self.method == 'GUE':
            perturbation = self.netG(image, label)
        if self.method == 'UEc':
            batch_noise = [] 
            for i in label:
                class_noise = self.noise[i]
                batch_noise.append(class_noise)
            perturbation = torch.stack(batch_noise).to(self.device)
        if self.method=='UE' or self.method=='TUE' or self.method == 'RUE':
            perturbation = self.noise[self.idx:self.idx+len(label)]
            self.idx += len(label)
        perturbation = torch.clamp(perturbation, -self.Epsilon, self.Epsilon)
        adv_image = perturbation + image
        adv_image = torch.clamp(adv_image, 0, 1)
        return adv_image
    
    def model_eval(self, model, epoch):
        self.model = model
        self.model.eval() # 停用dropout并在batchnorm层使用训练集的数据分布
        with torch.no_grad():
            writer.add_scalars('acc', {
                'dirty_trainset':self._model_eval(['dirty','trainset']),
                'dirty_testset':self._model_eval(['dirty','testset ']),
                'clean_trainset':self._model_eval(['clean','trainset']),
                'clean_testset':self._model_eval(['clean','testset '])
            }, epoch)
            
    def _model_eval(self,set=['clean','trainset']):
        '''测试代理模型在加噪/干净的训练/测试数据集上的准确率'''
        # num_samples = self.train_sum if set[1]=='trainset' else self.test_sum
        dataloader = self.train_dataloader if set[1]=='trainset' else self.test_dataloader
        num_correct = torch.zeros(self.num_classes).to(self.device)
        num_samples = torch.zeros(self.num_classes).to(self.device)
        num_correct_sum = 0
        self.idx = 0
        for _, data in enumerate(dataloader, 0):
            image, label = data
            image, label = image.to(self.device), label.to(self.device)
            input = image if set[0]=='clean' else self.add_noise(image, label)
            pred_lab = torch.argmax(self.model(input),1)
            mask=pred_lab==label
            for i in range(len(label)):
                num_correct[label[i]]+=mask[i]
                num_samples[label[i]]+=1
            num_correct_sum += torch.sum(pred_lab==label,0)
        acc = num_correct/num_samples
        acc_sum = num_correct_sum/torch.sum(num_samples) 
        time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        print(time_now,'model acc {} {} {} {:.2%}'.format(set[0],set[1],[round(i,2) for i in acc.tolist()],acc_sum))
        return acc_sum
    
    def loss_init(self): # 防止除零错误
        self.loss_perturb_sum = torch.tensor(1e-9).to(self.device)
        self.loss_ul_sum = torch.tensor(1e-9).to(self.device)
        self.num_itr = torch.tensor(1e-9).to(self.device)

    def train_noise(self,model, image, label):
        if self.method == 'GUE':
            self.train_GUE(model, image, label)
        if self.method == 'UEc':
            self.train_UEc(model, image, label)
        if self.method == 'UE':
            self.train_UE(model, image, label)
        if self.method == 'TUE':
            for _ in range(5):
                self.train_TUE(model, image, label)
        if self.method == 'RUE':
            self.train_RUE(model, image, label)
        self.idx += len(label)
    
    def train_GUE(self, model, image, label):
        '''更新GUE噪声'''
        # 对抗训练
        adv_per = torch.zeros_like(image).to(self.device) # 得到一个新tensor，内存、梯度分离(https://zhuanlan.zhihu.com/p/393041305)
        adv_per.requires_grad_(True)
        for _ in range(self.atk_step): 
            logits_adv = model(adv_per+image)
            loss_adv = F.cross_entropy(logits_adv, label)
            grad_adv = torch.autograd.grad(loss_adv, [adv_per])[0]
            with torch.no_grad(): # loss最大化，对抗训练
                adv_per.add_(torch.sign(grad_adv), alpha=self.atk_step_size)
                adv_per.clamp_(-self.atk_Epsilon, self.atk_Epsilon)
        # adv_per.grad.zero_()
        perturbation = self.netG(image+adv_per, label) # 生成器生成噪声
        # clipping trick，防止噪声过大
        adv_images = torch.clamp(perturbation, -self.Epsilon, self.Epsilon) + image
        adv_images = torch.clamp(adv_images, 0, 1)
        
        # 更新生成器
        self.optimizer_G.zero_grad()
        # 计算ul loss，目标模型的识别loss

        logits_dirty = model(self.trans(adv_images)) # [batch_size, 10]
        # logits_clean = model(self.trans(image)) 
        if self.netG.kmeans_label: # 使用k-means预测的label
            label = self.netG.lab.long()
        loss_dirty = F.cross_entropy(logits_dirty, label)
        # loss_clean = F.cross_entropy(logits_clean, label)
        loss_ul = loss_dirty# -0.01*loss_clean
        # 求噪声的平均大小，优化目标，噪声尽可能小
        # view(-1) -1表示一个不确定的数，不确定的地方可以写成-1；
        # perturbation.shape[0] 表示噪声的个数；dim=1 表示去掉dim=1的维度，在剩下的维度上求范数
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  # type: ignore

        # 总噪声loss 
        # if epoch<10 or loss_ul>0.5:
        loss_G = 100*loss_ul
        # else:
        #     loss_G = loss_ul + 0.001 * loss_perturb  
        loss_G.backward() #retain_graph=True)
        self.optimizer_G.step()

        self.loss_perturb_sum += loss_perturb
        self.loss_ul_sum += loss_ul
        self.num_itr += 1

    def train_UEc(self, model, image, label):
        '''更新UE噪声'''
        # x.data 返回和x相同tensor, 但不会加入到x的计算历史，不能被autograd追踪求微分
        # https://zhuanlan.zhihu.com/p/351687500 计算图(grad_fn)
        batch_noise = [] 
        for i in label:
            class_noise = self.noise[i].clone().detach()
            batch_noise.append(class_noise)
        perturbation = torch.stack(batch_noise).to(self.device)
        perturbation.requires_grad_(True) # 需要tensor的梯度
        perturb_img = image + perturbation
        logits = model(perturb_img)
        loss_ul = F.cross_entropy(logits, label)
        grad = torch.autograd.grad(loss_ul, [perturbation])[0] # 不会计算model的梯度
        # 样本噪声每步更新lr的长度，梯度只决定方向
        # a leaf Variable that requires grad 不能使用inplace操作
        with torch.no_grad():# 防止求loss_perturb时生成计算图
            perturbation.sub_(grad.data.sign(), alpha=self.lr_noise) # 若add_，导致loss增大（对抗训练）
            perturbation.clamp_(-self.Epsilon, self.Epsilon)
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  
            # 更新噪声
            class_noise = collections.defaultdict(list) # 为字典索引提供默认值
            for i in range(len(perturbation)): 
                # 若不使用item()，则每一个tensor都单独成list，因为值相同的tensor的id不相同
                class_noise[label[i].item()].append(perturbation[i].detach())
            for key in class_noise:
                self.noise[key] = torch.stack(class_noise[key]).mean(dim=0)

        self.loss_perturb_sum += loss_perturb
        self.loss_ul_sum += loss_ul
        self.num_itr += 1
    
    def train_UE(self, model, image, label):
        '''更新UE噪声'''
        # x.data 返回和x相同tensor, 但不会加入到x的计算历史，不能被autograd追踪求微分
        # https://zhuanlan.zhihu.com/p/351687500 计算图(grad_fn)
        perturbation = self.noise[self.idx:self.idx+len(label)].clone().detach() # 直接copy，后果是改变pert的同时改变noise
        perturbation.requires_grad_(True) # 需要tensor的梯度
        perturb_img = image + perturbation
        logits = model(perturb_img)
        loss_ul = F.cross_entropy(logits, label)
        grad = torch.autograd.grad(loss_ul, [perturbation])[0] # 不会计算model的梯度
        # 样本噪声每步更新lr的长度，梯度只决定方向
        # a leaf Variable that requires grad 不能使用inplace操作
        with torch.no_grad():# 防止求loss_perturb时生成计算图
            perturbation.sub_(grad.data.sign(),alpha=self.lr_noise) # 若add_，导致loss增大（对抗训练）
            perturbation.clamp_(-self.Epsilon, self.Epsilon)
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  
            # 更新噪声
            self.noise[self.idx:self.idx+len(label)] = perturbation.clone().detach()

        self.loss_perturb_sum += loss_perturb
        self.loss_ul_sum += loss_ul
        self.num_itr += 1

    def train_TUE(self, model, image, label):
        '''更新TUE噪声'''
        perturbation = self.noise[self.idx:self.idx+len(label)].clone().detach()
        perturbation.requires_grad_(True) # 需要tensor的梯度

        # SimSiam
        SimSiam_loss = self.SimSiam_net.k_grad_forward(image+perturbation)

        # CSD
        class_center = [] # 每个类的类中心 32*32*3=3027维，
        intra_class_dis = [] # 每个类的平均类内距
        c = torch.max(label) + 1 # num_classes
        for i in range(c): # 每个类求类中心和平均类内均
            idx_i = torch.where(label == i)[0]
            if idx_i.shape[0] == 0: # 这类样本数为0
                continue
            sample = perturbation.reshape(perturbation.shape[0], -1)
            class_i = sample[idx_i, :]
            class_i_center = class_i.mean(dim=0)
            class_center.append(class_i_center) 
            point_dis_to_center = torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))
            intra_class_dis.append(torch.mean(point_dis_to_center))
        if len(class_center) <= 1:
            return 0 # 这篇样本全是一个类
        class_center = torch.stack(class_center, dim=0)
        c = len(intra_class_dis)
        # p=2时，torch.cdist求向量之间的欧几里得距离，即类间距
        class_dis = torch.cdist(class_center, class_center, p=2) # TODO: this can be done for only one time in the whole set

        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1) # 去掉全0的对角线

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1) # 转置
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis # σi + σj
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1) # 去掉对角线i==j，只保留i!=j

        CSD_loss = ((intra_class_dis_pair_sum + 1e-5) / (class_dis + 1e-5)).mean()

        ul_loss = SimSiam_loss * 1 + CSD_loss * 0.1
        # perturbation.retain_grad()
        grad = torch.autograd.grad(ul_loss, [perturbation])[0] 
        # cluster_DB_loss.backward()
        with torch.no_grad():# 防止求loss_perturb时生成计算图
            perturbation.sub_(grad.data.sign(),alpha=self.lr_noise) # 若add_，导致loss增大（对抗训练）
            perturbation.clamp_(-self.Epsilon, self.Epsilon)
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  
            # 更新噪声
            self.noise[self.idx:self.idx+len(label)] = perturbation.detach()
        
        if torch.isnan(self.loss_ul_sum) or torch.isnan(ul_loss):
            print("nan")
        self.loss_perturb_sum += loss_perturb
        self.loss_ul_sum += ul_loss
        self.num_itr += 1

    def train_RUE(self, model, image, label):
        '''更新RUE噪声'''
        perturbation = self.noise[self.idx:self.idx+len(label)].clone().detach()
        perturbation.requires_grad_(True) # 需要tensor的梯度
        perturb_img = image + perturbation
        adv_per = perturbation.clone().detach() # 得到一个新tensor，内存、梯度分离(https://zhuanlan.zhihu.com/p/393041305)
        adv_per.requires_grad_(True)
        for _ in range(self.atk_step): 
            logits_adv = model(adv_per+image)
            loss_adv = F.cross_entropy(logits_adv, label)
            grad_adv = torch.autograd.grad(loss_adv, [adv_per])[0]
            with torch.no_grad(): # loss最大化，对抗训练
                adv_per.add_(torch.sign(grad_adv), alpha=self.atk_step_size)
                adv_per.clamp_(-self.atk_Epsilon, self.atk_Epsilon)
        logits = model(adv_per+image)
        loss_ul = F.cross_entropy(logits, label)
        grad = torch.autograd.grad(loss_ul, [adv_per])[0] 
        upd_lo = (perturb_img * grad).sum() # todo：这一步的目的
        grad = torch.autograd.grad(upd_lo, [perturbation])[0] 
        with torch.no_grad():
            perturbation.sub_(grad.data.sign(),alpha=self.lr_noise) # 若add_，导致loss增大（对抗训练）
            perturbation.clamp_(-self.Epsilon, self.Epsilon)
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  
            # 更新噪声
            self.noise[self.idx:self.idx+len(label)] = perturbation.clone().detach()
        
        self.loss_perturb_sum += loss_perturb
        self.loss_ul_sum += loss_ul
        self.num_itr += 1

    def print_loss(self, epoch):
        time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        print(time_now, "epoch %d: loss_perturb:%.3f, loss_ul:%.3f" %
                (epoch,self.loss_perturb_sum/self.num_itr, self.loss_ul_sum/self.num_itr))
        
    def save_noise(self):
        if self.method == 'GUE':
            Eps = str(int(self.atk_Epsilon*255))
            #file_name = './ul_models/'+self.datasetname+'_GUE'+str(self.size)+'t10.pth'
            file_name = './ul_models/'+self.datasetname+'GUE.pth'
            # file_name = './ul_models/'+self.datasetname+'_GUE'+str(self.size)+'a'+Eps+'.pth' # _e'+str(int(self.Epsilon*255))+'
            torch.save(self.netG.state_dict(), file_name)
        if self.method == 'UEc' or self.method == 'UE' or self.method == 'TUE':
            # file_name = './ul_models/'+self.datasetname+'_'+self.method+str(self.size)+'.pt'
            file_name = './ul_models/'+self.datasetname+self.method+'.pt'
            torch.save(self.noise, file_name)
        if self.method == 'RUE':
            Eps = str(int(self.atk_Epsilon*255))
            # file_name = './ul_models/'+self.datasetname+'_RUE'+str(self.size)+'a'+Eps+'.pt'
            file_name = './ul_models/'+self.datasetname+'RUE.pt'
            torch.save(self.noise, file_name)
        print(file_name,'saved!')

    def write_log(self, epoch, image_log, label_log):
        with torch.no_grad():
            rjust = lambda x:str(x).rjust(2,'0')
            checkpointspath = self.logpath+'/checkpoints/'
            if not os.path.exists(checkpointspath):
                os.mkdir(checkpointspath)
            imagespath = self.logpath+'/images/'
            if not os.path.exists(imagespath):
                os.mkdir(imagespath)
            if self.method == 'GUE':
                torch.save(self.netG.state_dict(), checkpointspath+rjust(epoch)+'.pth')
                if not os.path.exists(imagespath+'raw.png'):
                    torchvision.utils.save_image(image_log, imagespath+'raw.png',nrow=2)
                torchvision.utils.save_image(torch.clamp(self.netG(image_log,label_log), -self.Epsilon, self.Epsilon)*50, imagespath+rjust(epoch)+'.png',nrow=2)
            