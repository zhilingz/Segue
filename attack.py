import os
import time 
import torch
import torch.nn.functional as F
import collections
import models
import torchvision
import lpips
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class ul_Attack:
    def __init__(self,device,model,image_nc,Epsilon,lr1,lr2,target_class,gradient_base,class_wise,noise_prop,
    label_feature,train_dataloader,test_dataloader,epochs_train,epochs_test,generator_path,noise_path,face_label,kmeans_label):
        self.device = device
        self.model = model
        self.image_nc = image_nc
        self.Epsilon = Epsilon
        self.lr1 = lr1
        self.lr2 = lr2
        self.target_class = target_class
        self.gradient_base = gradient_base
        self.class_wise = class_wise
        self.noise_prop = noise_prop
        self.label_feature = label_feature
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs_train = epochs_train
        self.epochs_test = epochs_test
        self.generator_path = generator_path
        self.noise_path = noise_path
        self.face_label = face_label
        self.kmeans_label = kmeans_label
        self.train_itr = len(self.train_dataloader)
        self.train_sum = len(self.train_dataloader.dataset)
        self.test_sum = len(self.test_dataloader.dataset)
        self.num_classes = len(self.train_dataloader.dataset.classes)

    def image_quality(self):
        self.idx = 0
        mse_sum = 0
        lpips_sum = 0
        lpips_loss = lpips.LPIPS().to(self.device) # net='vgg'
        for _, data in enumerate(self.train_dataloader):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            adv_images = self.add_noise(images,labels)
            mse_sum += F.mse_loss(images,adv_images)
            lpips_sum += lpips_loss.forward(images,adv_images)
        print('Image quality between clean and dirty:\nMSE: %.6f\nLPIPS: %.6f'%(
            mse_sum/self.train_itr, lpips_sum.mean()/self.train_itr))
        
    def noise_expend(self, labels):
        '''将噪声的尺寸按标签扩展，从[10,...]扩展到[bs,...]'''
        batch_noise = [] 
        for label in labels:
            class_noise = self.noise[label]
            batch_noise.append(class_noise)
        return torch.stack(batch_noise).to(self.device)

    def add_noise(self,train_images,train_label):
        '''根据gradient_base与class_wise两个参数来选择加噪方式'''
        # 以noise_prop的概率添加噪声，保证训练集中毒比例为noise_prop
        if train_label[0] < self.noise_prop*10: 
            if self.gradient_base and self.class_wise:
                perturbation = self.noise_expend(train_label)
            if self.gradient_base and not self.class_wise:
                perturbation = self.noise[self.idx:self.idx+len(train_label)]
                self.idx += len(train_label)
            if not self.gradient_base:
                perturbation = self.netG(train_images, train_label)
            perturbation = torch.clamp(perturbation, -self.Epsilon, self.Epsilon)
            if type(self.target_class)==int: # 保护特定类，其余的类噪声都变为0向量
                mask = train_label==self.target_class
                perturbation *= mask.unsqueeze(1).expand(16,3).unsqueeze(2).expand(16,3,32).unsqueeze(2).expand(16,3,32,32)
            adv_images = perturbation + train_images
            adv_images = torch.clamp(adv_images, 0, 1)
        else:
            adv_images = train_images
        # adv_images = torchvision.transforms.RandomCrop(224, padding=40)(adv_images)
        # adv_images = torchvision.transforms.RandomRotation(20)(adv_images)
        return adv_images

    def model_train(self, train_images, train_label, ):
        '''训练一个batchsize的代理模型'''
        adv_images = self.add_noise(train_images,train_label)
        logits_model = self.model(adv_images)
        if self.train_dataloader.dataset.root == 'dataset/WebFace10/train_clean':
            loss_model = torch.nn.NLLLoss()(F.log_softmax(logits_model, dim = -1), train_label)
        else:
            loss_model = F.cross_entropy(logits_model, train_label)
        self.optimizer_model.zero_grad()
        loss_model.backward()
        self.optimizer_model.step()
        self.scheduler.step()
        return loss_model

    def model_eval(self, epoch):
        with torch.no_grad():
            if self.class_wise or not self.gradient_base: # class_wise的噪声加在测试集上
                writer.add_scalars('acc', {
                    'dirty_trainset':self._model_eval(['dirty','trainset']),
                    'dirty_testset':self._model_eval(['dirty','testset ']),
                    'clean_trainset':self._model_eval(['clean','trainset']),
                    'clean_testset':self._model_eval(['clean','testset '])
                }, epoch)
            else:
                writer.add_scalars('acc', {
                    'dirty_trainset':self._model_eval(['dirty','trainset']),
                    'clean_trainset':self._model_eval(['clean','trainset']),
                    'clean_testset':self._model_eval(['clean','testset '])
                }, epoch)

    def _model_eval(self,set=['clean','trainset']):
        '''测试代理模型在加噪/干净的训练/测试数据集上的准确率'''
        # num_samples = self.train_sum if set[1]=='trainset' else self.test_sum
        dataloader = self.train_dataloader if set[1]=='trainset' else self.test_dataloader
        num_correct = torch.zeros(10).to(self.device)
        num_samples = torch.zeros(10).to(self.device)
        num_correct_sum = 0
        self.idx = 0
        self.model.eval() # 停用dropout并在batchnorm层使用训练集的数据分布
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
        print(time_now,'Agent model accuracy of {} {}: {} {:.2%}'.format(set[0],set[1],[round(i,2) for i in acc.tolist()],acc_sum))
        return acc_sum

    def train_noise(self, x, labels):
        '''基于梯度的方法,用一个batchsize更新噪声'''
        perturbation = self.noise[self.idx:self.idx+len(labels)]
        # x.data 返回和x相同tensor, 但不会加入到x的计算历史，不能被autograd追踪求微分
        perturb_img = x.data + torch.clamp(perturbation, -self.Epsilon, self.Epsilon)
        perturb_img = torch.autograd.Variable(torch.clamp(perturb_img,0,1), requires_grad=True)
        opt = torch.optim.SGD([perturb_img], lr=1e-3)
        opt.zero_grad()
        self.model.zero_grad()
        logits = self.model(perturb_img)
        loss_ul = F.cross_entropy(logits, labels)
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  
        perturb_img.retain_grad()
        loss_ul.backward()
        # 样本噪声每步更新0.05的长度，梯度只决定方向
        perturb_img = perturb_img.data - perturb_img.grad.data.sign()*self.lr1
        perturbation = torch.clamp(perturb_img.data - x.data, -self.Epsilon, self.Epsilon)
        if self.class_wise: # 同一类的所有样本上的噪声累加作为类噪声
            class_noise = collections.defaultdict(list)
            for i in range(len(perturbation)):
                class_noise[labels[i].item()].append(perturbation[i].detach().cpu())
            for key in class_noise:
                self.noise[key] = torch.stack(class_noise[key]).mean(dim=0)
        else: # 每个样本的噪声单独更新
            self.noise[self.idx:self.idx+len(labels)] = perturbation
            self.idx += len(labels) 
        return loss_perturb, loss_ul

    def train_G(self, x, labels):
        '''基于优化的方法,用一个batchsize更新G'''
        # 更新判别器
        perturbation = self.netG(x, labels) # 生成器生成噪声
        # clipping trick，防止噪声过大
        adv_images = torch.clamp(perturbation, -self.Epsilon, self.Epsilon) + x
        adv_images = torch.clamp(adv_images, 0, 1)

        # 更新生成器
        self.optimizer_G.zero_grad()
        # 计算ul loss，目标模型的识别loss
        logits_model = self.model(adv_images) # [batch_size, 10] 
        loss_ul = F.cross_entropy(logits_model, labels)
        # 求噪声的平均大小，优化目标，噪声尽可能小
        # view(-1) -1表示一个不确定的数，不确定的地方可以写成-1；
        # perturbation.shape[0] 表示噪声的个数；dim=1 表示去掉dim=1的维度，在剩下的维度上求范数
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  # type: ignore

        # 总噪声loss 
        loss_G = loss_ul + 0.001 * loss_perturb  
        loss_G.backward() #retain_graph=True)
        self.optimizer_G.step()

        return loss_perturb, loss_ul 

    def train(self):
        if not self.gradient_base:
            self.netG = models.Generator(self.image_nc,self.image_nc,self.label_feature,self.num_classes, self.face_label,self.kmeans_label).to(self.device)
            # initialize all weights
            self.netG.apply(models.weights_init)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=self.lr1)
        elif self.class_wise: # 随机初始化10个类的噪声，perturbation的尺寸为[bs,3,32,32]
            self.noise = torch.FloatTensor(*[10,3,32,32]).uniform_(-0.05, 0.05).to(self.device)
        else: # sample_wise的梯度方法，噪声的个数和训练集数据个数对应
            self.noise = torch.FloatTensor(*[self.train_sum,3,32,32]).uniform_(-0.05, 0.05).to(self.device)
        self.optimizer_model = torch.optim.Adam(self.model.parameters(), lr=self.lr2)
        self.scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer_model, T_max =  self.epochs_train)

        self.model_eval(0)
        for epoch in range(1, self.epochs_train+1):
            loss_perturb_sum = 0
            loss_ul_sum = 0
            if epoch%5 == 1: # 隔几个epoch更新一次目标模型
                self.model.train()
                self.idx = 0
                for _, data in enumerate(self.train_dataloader, start=0):
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.model_train(images, labels)
                self.model_eval(epoch)
            
            # 基于梯度
            if self.gradient_base: 
                if epoch%10 == 0: # 更新学习率
                    print('lr:',self.optimizer_model.param_groups[0]['lr']) # 打印学习率
                    # self.lr2 = self.lr2*0.5
                    # self.optimizer_model = torch.optim.Adam(self.model.parameters(),lr=self.lr2)

                self.idx = 0
                for _, data in enumerate(self.train_dataloader): # 更新噪声
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    loss_perturb_batch, loss_ul_batch = self.train_noise(images, labels)
                    loss_perturb_sum += loss_perturb_batch
                    loss_ul_sum += loss_ul_batch

                if epoch%20==0: # 保存
                    if self.class_wise: # class_wise梯度式不可学习样本
                        pert_file = './ul_models/class_perturbation_epoch_'+str(epoch)+'.pt'
                    else:
                        pert_file = './ul_models/sample_perturbation_epoch_'+str(epoch)+'.pt'
                    torch.save(self.noise, pert_file)
                    print(pert_file,'saved!')
            
            # 基于G
            if not self.gradient_base: 
                if epoch%10 == 0: # 更新学习率 
                    self.lr1 = self.lr1*0.5
                    self.lr2 = self.lr2*0.5
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=self.lr1)
                    self.optimizer_model = torch.optim.Adam(self.model.parameters(),lr=self.lr2)
                
                for _, data in enumerate(self.train_dataloader): # 更新G
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    loss_perturb_batch, loss_ul_batch = self.train_G(images, labels)
                    loss_perturb_sum += loss_perturb_batch
                    loss_ul_sum += loss_ul_batch

                if epoch%10==0:# 保存
                    # if self.kmeans_label:
                    #     netG_file_name = './ul_models/netG_kmeanslabel.pth' # _e'+str(int(self.Epsilon*255))+'
                    torch.save(self.netG.state_dict(), self.generator_path)
                    print(self.generator_path,'saved!')

            # print statistics
            time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            print(time_now, "epoch %d: loss_perturb:%.3f, loss_ul:%.3f" %
                (epoch, loss_perturb_sum/self.train_itr, loss_ul_sum/self.train_itr))
        

    def test(self): 
        if not self.gradient_base:
            self.netG = models.Generator(self.image_nc,self.image_nc,self.label_feature,self.num_classes,self.face_label).to(self.device)
            self.netG.load_state_dict(torch.load(self.generator_path))
            print(self.generator_path, 'load!')
        elif self.class_wise: # 直接用随机噪声，效果也很好
            self.noise = torch.FloatTensor(*[10,3,224,224]).uniform_(-0.05, 0.05).to(self.device)
            # self.noise.clamp_(0,self.Epsilon)
            print('Random classwise noise init!')
        else:
            self.noise = torch.load(self.noise_path)
            print(self.noise_path, 'load!')
        self.optimizer_model = torch.optim.Adam(self.model.parameters(), lr=self.lr2)
        self.scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer_model, T_max =  self.epochs_test)

        # self.image_quality()
        self.model_eval(0)
        for epoch in range(1, self.epochs_test+1):
            self.idx = 0
            loss_sum = 0
            if epoch %1 == 0: # 降低学习率，防止振荡
                print('lr:',self.optimizer_model.param_groups[0]['lr']) # 打印学习率
            #     self.lr2 = self.lr2 * 0.9
            #     # self.optimizer_model = torch.optim.SGD(model.parameters(), self.lr2=self.lr2, momentum=0.9, weight_decay=5e-4)
            #     self.optimizer_model = torch.optim.Adam(self.model.parameters(), lr=self.lr2)
            #     print('lr=%.5f'%(self.lr2))
            self.model.train()
            for _, data in enumerate(self.train_dataloader):
                # 取一批数据，生成对应不可学习噪声
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                loss_sum += self.model_train(images, labels)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            print(time_now, 'Loss in Epoch %d: %.4f' % (epoch, loss_sum/self.train_itr))
            
            self.model_eval(epoch)
            
            if loss_sum/self.train_itr<0.0001:
                break

        writer.close()
        return
        