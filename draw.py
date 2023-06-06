import matplotlib
import numpy as np
from  matplotlib import pyplot as plt


#设置图片的大小
matplotlib.rc('figure', figsize = (9, 5)) #单位为厘米
#设置字体的大小
matplotlib.rc('font', size = 14) #size为字体的大小
#是否显示背景网格
matplotlib.rc('axes', grid = False)
#grid：取为Flase为不显示背景网格，True为显示
#背景颜色
matplotlib.rc('axes', facecolor = 'white')
#白色：white
#绿色：green
#黄色：yellow
#黑色：black
#灰色：grey
x1=np.array([
    4.075,
    3.836,
    2.294,
    1.546,
    1.436,
    2.174,
    2.445,
    1.917,
    0.886,
    0.575,
])
x2=np.array([
    4.000,
    2.695,
    1.898,
    1.106,
    1.092,
    0.899,
    0.708,
    0.261,
    0.239,
    0.217,
])
x3=np.array([
    4.044,
    1.933,
    1.069,
    0.144,
    0.056,
    0.013,
    0.009,
    0.007,
    0.005,
    0.004,
])
y=np.array([int(2*x) for x in range(0,10)])
plt.plot(y,x1,color='red',marker='*',label='w/o side-information')
plt.plot(y,x2,color='m',marker='*',label='w/ pseudo-label')
plt.plot(y,x3,color='blue',marker='x',label='w/ true-label')
plt.xticks(y)
plt.xlabel('Epochs')
plt.ylabel('Training Loss of Surrogate Model')
plt.legend(loc=1)
plt.savefig('images/label guide.png')