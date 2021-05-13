import os
import shutil
import numpy as np
import time

datasetPath = 'E:\cfy\deepLearning\dataset\snow leopard/'

time_start = time.time()

# 文件路径

dataset_len = 1424  #其中猫与狗的图片数量各占一半
# index = np.arange(dataset_len)
# np.random.shuffle(index)#打断顺序
# train_len = int(len(index)*0.8)
# index_train = index[:train_len] #取前len（index）的80%作为index2
# index_test = index[train_len:]

f1 = open('E:\cfy\deepLearning\dataset\snow leopard\original\images_lable.txt','r')
f2 = open('E:\cfy\deepLearning\dataset\snow leopard\original\\train.txt','w')
f3 = open('E:\cfy\deepLearning\dataset\snow leopard\original\\test.txt','w')

originalImagesList = f1.readlines()
np.random.shuffle(originalImagesList)

train_len = int(dataset_len*0.8)
for i in range(train_len):
    # originalImagesList[i] = originalImagesList[i].strip()
    f2.write('{}'.format(originalImagesList[i]))
for i in range(train_len,dataset_len):
    # originalImagesList[i] = originalImagesList[i].strip()
    f3.write('{}'.format(originalImagesList[i]))

f1.close()
f2.close()
f3.close()
time_end = time.time()
print('雪豹训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))

