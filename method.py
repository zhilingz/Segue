import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import Generator
from models import weights_init

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class Method():
    def __init__(self,train_dataloader,test_dataloader,datasetname,modelname,num_classes,size,Epsilon,kmeans_label,rho,device):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.datasetname = datasetname
        self.modelname = modelname
        self.num_classes = num_classes
        self.Epsilon = Epsilon
        self.kmeans_label = kmeans_label
        self.size = size
        self.device = device
        self.lr_noise = 5e-4
        self.atk_Epsilon = rho/255 # 对抗训练噪声的强度应该小于不可学习噪声强度
        self.atk_step_size = self.atk_Epsilon/5 # 对抗训练更新大小
        self.atk_step = 5 
        self.netG = Generator(3,3,num_classes,kmeans_label,datasetname).to(device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=self.lr_noise)
        self.model_init(kmeans_label)
        self.trans = transforms.Compose([
                transforms.GaussianBlur(kernel_size=(3,3), sigma=0.2),
                transforms.RandomAdjustSharpness(2,1),
                # # transforms.RandomCrop(224, padding=2),
                transforms.RandomHorizontalFlip(0.1),
                transforms.RandomVerticalFlip(0.1)
                ])
        print('kmeans_label:{}'.format(kmeans_label))
        print('distortion layer')
        print(self.trans)
        print('lr_noise: {:.4e}'.format(self.lr_noise))

    def model_init(self, kmeans_label):
        self.netG.apply(weights_init)
        if kmeans_label: # 不需要类别信息
            model_path = 'nets/facenet_mobilenet.pth' # 使用其他人脸数据集上预训练的人脸特征提取器
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
        with torch.no_grad():
            perturbation = self.netG(image, label)
        perturbation = torch.clamp(perturbation, -self.Epsilon, self.Epsilon)
        adv_image = perturbation + image
        adv_image = torch.clamp(adv_image, 0, 1)
        return adv_image
    
    def model_eval(self, model, epoch):
        self.model = model
        self.model.eval() # 停用dropout并在batchnorm层使用训练集的数据分布
        with torch.no_grad():
            self._model_eval(['dirty','trainset'])
            self._model_eval(['dirty','testset '])
            self._model_eval(['clean','trainset'])
            self._model_eval(['clean','testset '])
            
    def _model_eval(self,set=['clean','trainset']):
        '''测试代理模型在加噪/干净的训练/测试数据集上的准确率'''
        # num_samples = self.train_sum if set[1]=='trainset' else self.test_sum
        dataloader = self.train_dataloader if set[1]=='trainset' else self.test_dataloader
        num_correct = torch.zeros(self.num_classes).to(self.device)
        num_samples = torch.zeros(self.num_classes).to(self.device)
        num_correct_sum = 0
        for _, data in enumerate(dataloader, 0):
            image, label = data
            image, label = image.to(self.device), label.to(self.device)
            if self.kmeans_label:
                with torch.no_grad():
                    features = self.netG.facenet.feature_extract(image).detach()
                    lab = self.netG.kmeans.predict(features.cpu())
                label = torch.from_numpy(lab).cuda().long()
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

    def train_noise(self, model, image, label):
        '''更新噪声'''
        # freeze model, update G
        for param in model.parameters():
            param.requires_grad = False
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
        logits = model(self.trans(adv_images)) # [batch_size, 10]
        loss_ul = F.cross_entropy(logits, label)
        
        # 求噪声的平均大小，优化目标，噪声尽可能小
        # view(-1) -1表示一个不确定的数，不确定的地方可以写成-1；
        # perturbation.shape[0] 表示噪声的个数；dim=1 表示去掉dim=1的维度，在剩下的维度上求范数
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  # type: ignore

        loss_G = loss_ul + 0.001 * loss_perturb   # loss_perturb在100以内，更换权重会影响结果
        loss_G.backward() #retain_graph=True)
        self.optimizer_G.step()

        self.loss_perturb_sum += loss_perturb
        self.loss_ul_sum += loss_ul
        self.num_itr += 1

        
        for param in model.parameters():
            param.requires_grad = True

    def print_loss(self, epoch):
        time_now = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        print(time_now, "epoch %d: loss_perturb:%.3f, loss_ul:%.3f" %
                (epoch,self.loss_perturb_sum/self.num_itr, self.loss_ul_sum/self.num_itr))
        
    def save_G(self):
        file_name = './ul_models/'+self.datasetname+'_G.pth'
        torch.save(self.netG.state_dict(), file_name)
        print(file_name,'saved!')

