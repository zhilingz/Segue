from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Mixup_data(img, label, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda
    lamda: 融合比例.lamda为1时不融合
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_img = lam * img + (1 - lam) * img.flip(dims=(0, ))
    label_a, label_b = label, label.flip(dims=(0, ))
    return mixed_img, label_a, label_b, lam

def Cutmix_data(img, label, n_holes, length):
    '''batch中第i个图像和第(batchsize-i)个图像Cutmix
    当用2个112*112的块来Cutmix时,最后loss值趋于1.6698
    loss_fun(torch.tensor([[1.0,1,0,0,0,0,0,0,0,0,0]]),torch.tensor([0]))=1.6698
    '''
    if n_holes==0:
        return img, label, label.flip(dims=(0, )), 1
    b , _, h, w = img.shape
    mask = np.ones((h, w), np.float32)
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0
    mixed_img = np.transpose(img.numpy(),(0,2,3,1))
    img_copy = mixed_img.copy()
    for idx in range(b):
        mixed_img[idx][mask==0] = img_copy[b-idx-1][mask==0]
    return torch.from_numpy(np.transpose(mixed_img, (0,3,1,2))), label, label.flip(dims=(0, )), np.sum(mask==1)/(h*w)

def mix_criterion(criterion, logits, label_a, label_b, lam):
    return lam * criterion(logits, label_a) + (1 - lam) * criterion(logits, label_b)

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        p (float): 概率值,依概率执行该操作
    """
    def __init__(self, n_holes=1, length=16, p=1.0):
        self.n_holes = n_holes
        self.length = length
        self.p = p
 
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image, size (H, W, C).
        Returns:
            PIL Image: PIL image with n_holes of dimension length x length cut out of it.
        """
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            mask = np.ones((h, w), np.float32)
            for _ in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0
            img[mask == 0] = [0,0,0]
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(n_holes={0}, length={1}, p={2})'.format(self.n_holes, self.length, self.p)

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr(float): Signal Noise Rate,越小噪声越多,range(0,1)
        p(float): 概率值,依概率执行该操作
    """
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, -1), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2) # 复制至另外两通道
            img[mask == 1 ] = 255   # 盐噪声
            img[mask == -1] = 0     # 椒噪声
            return Image.fromarray(img.astype('uint8')).convert('RGB')
        else:
            return img
            
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(snr={0}, p={1})'.format(self.snr, self.p)

def transform_view():
    orig_img = Image.open('/data/zhangzhiling/WebFace10/trainGUE/0/n0.png')
    transform = T.Compose([
        T.GaussianBlur(kernel_size=9, sigma=6),
        # T.RandomAdjustSharpness(5,1),
        # T.RandomRotation(200),
        # Cutout(2, 100),
        # AddPepperNoise(0.5,1.0),
        T.ToTensor(),
        ])
    torchvision.utils.save_image(transform(orig_img), 'images/view.png')

def TNSE_view():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from models import Facenet
    test_transforms = T.Compose([
        T.Resize([224,224]),
        T.ToTensor(),
        ])
    train_dataset = torchvision.datasets.ImageFolder('dataset/WebFace10/train_clean',test_transforms)
    train_dataloader = DataLoader(train_dataset,batch_size=130,shuffle=True,num_workers=4,drop_last=False)
    model = Facenet(backbone='mobilenet', num_classes=len(train_dataset.classes)).to(device)
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('pre_models/facenet_mobilenet.pth', map_location = device)
    # load_key, no_load_key, temp_dict = [], [], {}
    # for k, v in pretrained_dict.items():
    #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
    #         temp_dict[k] = v
    #         load_key.append(k)
    #     else:
    #         no_load_key.append(k)
    # model_dict.update(temp_dict)
    # model.load_state_dict(model_dict)
    plt.figure(figsize=(12, 12))
    tsne = TSNE(n_components=2, perplexity=10,init='pca',learning_rate='auto')
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
    feature = torch.zeros(1300,128)
    labels = torch.zeros(1300)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            images, label = data
            feature[i:130+i] = model.feature_extract(images.to(device)).cpu()
            labels[i:130+i] = label
    feature = feature.cpu()
    data_2d = tsne.fit_transform(feature.view(1300,-1)[:130])
    for i in range(10):
        plt.scatter(data_2d[labels[:130] == i, 0], data_2d[labels[:130] == i, 1], c=colours[i], label=i)
    plt.legend()
    plt.savefig('images/tnse130_nopretrain.jpg')

def JPEG():
    '''https://www.yisu.com/zixun/619758.html''' 
    # img = torch.FloatTensor(*[3,224,224]).uniform_(0, 1)
    img = Image.open('images/002.jpg')
    img.convert('RGB')
    img = T.ToTensor()(img)
    # torchvision.utils.save_image
    # torch.float->torch.uint8 都是向下取整，所以要加0.5
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    for level in range(1,10):
        im.save('images/test'+str(level)+'.png',quality=level*10)#compress_level=level) # jpg默认95左右
    for level in range(1,10):
        im.save('images/test'+str(level)+'.jpg',quality=level*10)#compress_level=level)
    im.save('images/test0.png')
    im.save('images/test0.jpg')
    tensor_png = []
    tensor_jpg = []
    for level in range(0,10):
        reimg_png = Image.open('images/test'+str(level)+'.png')
        reimg_jpg = Image.open('images/test'+str(level)+'.jpg')
        reimg_png.convert('RGB')
        reimg_jpg.convert('RGB')
        tensor_png.append(T.ToTensor()(reimg_png))
        tensor_jpg.append(T.ToTensor()(reimg_jpg))
    print('end')

if __name__=='__main__':
    transform_view()

    
