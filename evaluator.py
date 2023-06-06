import time
import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class evaluator():
    def __init__(self,train_dataloader, test_dataloader, add_noise, num_classes, device):
        self.device = device
        self.train_dataloader= train_dataloader
        self.test_dataloader = test_dataloader
        self.add_noise = add_noise
        self.num_classes = num_classes

    def model_eval(self, model, epoch):
        self.model = model
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