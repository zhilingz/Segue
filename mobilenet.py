import torch.nn as nn

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )
    
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        # nn.ZeroPad2d([0,1,0,1]) if stride == 2 else nn.ZeroPad2d([1,1,1,1]),
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(3, 32, 2), 
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1), 

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stage1(x) # [16, 3, 224, 224] -> [16, 256, 33, 33]
        x = self.stage2(x) # ->[16, 512, 19, 19]
        x = self.stage3(x) # ->[16, 1024, 12, 12]
        x = self.avg(x)    # ->[16, 1024, 1, 1]
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x