import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from makeDataset import *
from build_model import *
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
import time
from utils import accuracy, AverageMeter, save_checkpoint
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
rootPath = 'E:\cfy\deepLearning\dataset\snow leopard'

def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224,224), 'white')
    return img

class RandomDataset(Dataset): #产生测试集/验证集
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        # with open(rootPath+'\original/hard_samples/test3.txt', 'r') as fid:
        with open(rootPath + '/all/test1.txt', 'r') as fid:
                self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split(',,,')
        # image_path = image_name
        image_path = 'E:\cfy\deepLearning/dataset' + image_name.split('dataset')[-1]
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label,image_path]


    def __len__(self):
        return len(self.imglist)

def main():
    time_start = time.time()

    batchsize = 16
    namePostfix = 'focal_50_0.4_1000'

    # 定义两个数组
    test_Loss_list = []
    test_Accuracy_list = []
    # net = ResNeSt().to(device)
    net = ResNeSt()
    test_model = './models_pkl/model_best_3_3_'  + namePostfix +  '.pth.tar'
    ckpt = torch.load(test_model, map_location="cpu")
    # net.load_state_dict(ckpt["state_dict"])
    net.load_state_dict(ckpt["state_dict"])
    print('model_epoch:',ckpt["epoch"],"test_prec:",ckpt['best_prec1'])
    net = net.cuda()
    # net = net
    criterion = nn.CrossEntropyLoss()
    # imageResize = 512
    imageResize = 448

    test_dataset = RandomDataset(transform=transforms.Compose([
        transforms.Resize([imageResize, imageResize]),
        # transforms.CenterCrop([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=False,
        num_workers=4, pin_memory=True)
    Accuracy,Precision,Recall = tests11(test_loader,net,criterion,namePostfix)
    print('测试集准确率：'+str(Accuracy) + '| 精准率：'+str(Precision) + '| 召回率：'+str(Recall))
    time_end = time.time()
    print('雪豹训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))




def tests11(test_loader, net, criterion,namePostfix):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()

    # top5 = AverageMeter()
    #
    # # switch to evaluate mode
    # model.eval()
    end = time.time()
    net.eval()
    # net = ResNeSt().to(device)
    # f1 = open('F:\SoftwareProgram\Data\dataset\snow leopard\original\\test_negative.txt', 'w')
    f1 = open('./test_negative3_1_'  + namePostfix +  '.txt', 'w')
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for i, (inputs, targets,image_paths) in enumerate(test_loader):
            # try:

            inputs = inputs.cuda()
            # inputs = inputs
            targets = targets.long().cuda().squeeze()  # 对变量维度进行压缩
            # targets = targets.long().squeeze()  # 对变量维度进行压缩

            # compute output
            logits = net(inputs)
            loss = criterion(logits, targets)
            # except:
            #     print(image_paths+'\n\n\n')

            #logits.topk(6, 1, True, True)

            prec1,correct= accuracy(logits, targets, 1)
            for index,prec_index in enumerate(correct):
                if prec_index == True and targets[index] == 1:
                    TP += 1
                if prec_index == False:
                    f1.write('{},,,{}\n'.format(image_paths[index],targets[index]))   #输出预测错误的图像到test_negative.txt
                    if targets[index] == 0:
                        FP += 1
                    if targets[index] == 1:
                        FN += 1

            # prec5 = accuracy(logits, targets, 5)
            softmax_losses.update(loss.item(), logits.size(0))
            top1.update(prec1, logits.size(0))
            # top5.update(prec5, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % 1 == 0:
                print('Time: {time}\nTest: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(test_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                        top1=top1, time=time.asctime(time.localtime(time.time()))))

        # test_Loss_list.append(softmax_losses.avg)
        # test_Accuracy_list.append(100 * top1.avg)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    f1.close()
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    # Recall = 0
    return top1.avg,Precision*100,Recall*100

if __name__ == '__main__':
    main()