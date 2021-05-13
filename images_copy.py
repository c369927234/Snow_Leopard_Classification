import shutil
import numpy as np
rootPath = 'E:\cfy\deepLearning\dataset\snow leopard'
# test_negative_pathList = rootPath+'\original/hard_samples/test3.txt'
namePostfix = 'focal_50_0.4_1000_640'
test_negative_pathList = 'test_negative3_1_'  + namePostfix +  '.txt'
with open(test_negative_pathList, 'r') as f:
    test_negative_imageName = [x.strip().split(',,,')[0] for x in f]
np.random.shuffle(test_negative_imageName)
# idx = 0
for target in test_negative_imageName:
    # shutil.copy('F:\Dataset\ACCV_Datesets/test/' + target, './data/prec_negative_imgs/')
    shutil.copy(target, './data/prec_negative_imgs/focal_0.4_1000_640/model3_1/')
    # idx+=1
    # if idx == 100: break





# shutil.copy('F:\Dataset\ACCV_Datesets/test/' + target, './data/test')