import shutil
import os
# id_list = [0]*10178 # 一个id对应一个list元素
root = '/data/zhangzhiling/celeba/img_align_celeba/'
new_root = '/data/zhangzhiling/CelebA10/raw/'
id_index=['14','15','17','21','22','23','25','31','32','38'] # 有30张人脸图片的id
if not os.path.exists(new_root):
    os.mkdir(new_root)
    for i in id_index:
        os.mkdir(new_root+i)
with open('/data/zhangzhiling/celeba/identity_CelebA.txt','r') as f:
    for line in f.readlines():
        line_split = line.split()
        # id_list[int(line_split[1])]+=1
        if line_split[1] in id_index:
            shutil.copy(root+line_split[0],new_root+line_split[1]+'/'+line_split[0])
print('end')