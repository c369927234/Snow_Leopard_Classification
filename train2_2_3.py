
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms
from torchsummary import summary
from PIL import Image
from build_model import *
# from makeDataset import *
from resnest.torch import resnest50,resnest50_fast_2s1x64d,resnest101
import visdom
from utils import accuracy, AverageMeter, save_checkpoint
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from focalLoss import focal_loss



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


rootPath = 'E:\cfy\deepLearning\dataset\snow leopard'

def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error_2.txt', 'a') as fid:
            fid.write(path + ',,,' + time.asctime(time.localtime(time.time())) + '\n')
        return Image.new('RGB', (224,224), 'white')
    return img

class RandomDataset2(Dataset): #产生测试集/验证集
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        with open(rootPath+'/all/test2.txt', 'r') as fid:
            self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split(',,,')
        image_path = 'E:\cfy\deepLearning/dataset' + image_name.split('dataset')[-1]
        # image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label,image_path]


    def __len__(self):
        return len(self.imglist)

class RandomDataset3(Dataset): #产生测试集/验证集
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        with open(rootPath+'/all/test3.txt', 'r') as fid:
            self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split(',,,')
        image_path = 'E:\cfy\deepLearning/dataset' + image_name.split('dataset')[-1]
        # image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label,image_path]


    def __len__(self):
        return len(self.imglist)

class BatchDataset(Dataset): #产生训练集
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        with open(rootPath+'/all/train2.txt', 'r') as fid:
            self.imglist = fid.readlines()

        self.labels = []
        for line in self.imglist:
            image_path, label = line.strip().split(',,,')
            self.labels.append(int(label))
        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)


    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split(',,,')
        image_path = 'E:\cfy\deepLearning/dataset' + image_name.split('dataset')[-1]
        # image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label]



    def __len__(self):
        return len(self.imglist)


def main():
    epoch = 50
    # lr = 0.01
    # lr = 0.001
    batchsize = 16
    # 定义两个数组
    train_Loss_list = []
    train_Accuracy_list = []
    test_Loss_list2 = []
    test_Accuracy_list2 = []
    test_Loss_list3 = []
    test_Accuracy_list3 = []
    namePostfix = 'focal_50_0.4_1000'
    # imageResize = 640
    imageResize = 512

    # imageRandomCrop = 512
    imageRandomCrop = 448


    train_dataset = BatchDataset(transform=transforms.Compose([
        transforms.Resize([imageResize, imageResize]),
        transforms.RandomCrop([imageRandomCrop, imageRandomCrop]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=4, batch_size=batchsize, pin_memory=True, shuffle=True)

    net = ResNeSt().to(device)
    # test_model = "./models_pkl/checkpoint_relu_fc2.pth.tar"
    # ckpt = torch.load(test_model, map_location="cpu")
    # net.load_state_dict(ckpt["state_dict"])



    # for i in net.children():
    #     print(i)
    # net2 = resnest50(pretrained=False).to(device)
    # img = Image.open("D:\ScienceResearch\SoftwareProgram\Data\dataset\CUB_200_2011\images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
    # x = transforms.Resize([32,32])(img)
    # x = transforms.ToTensor()(x)#用于将Image读取的img转换为tensor
    # x = x.view(-1,x.size(0),x.size(1),x.size(2)).cuda()
    # out = net(x)

    # summary(net2,(3,512,512))

    # criterion = nn.CrossEntropyLoss()
    criterion = focal_loss()

    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # is_vis = False  # 是否进行可视化，如果没有visdom可以将其设置为false
    # if is_vis:
    #     vis = visdom.Visdom()
    #     viswin1 = vis.line(np.array([0.]), np.array([0.]),
    #                        opts=dict(title="Loss/Step", xlabel="100*step", ylabel="Loss"))

    best_prec1 = 0

    for e in range(epoch):
        # train_loss = []
        batch_time = AverageMeter()
        softmax_losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()

        net.train()
        # f1 = open('F:\SoftwareProgram\Data\dataset\snow leopard\original\\tarin_negative.txt', 'w')
        # yl = torch.Tensor([0]).cuda()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = targets.long().cuda().squeeze() #对变量维度进行压缩

            optimizer.zero_grad()

            pred = net(inputs)
            loss = criterion(pred, targets)

            prec1,correct = accuracy(pred, targets, 1)
            # for prec_index in correct:
            #     if prec_index == False:
            #         f1.write('{}'.format(image_paths[prec_index]))   #输出预测错误的图像到train_negative.txt
            # prec5 = accuracy(logits, targets, 5)

            softmax_losses.update(loss.item(), pred.size(0)) #item()将只有一个元素的numpy数组或tensor张量转化为标量
            top1.update(prec1, pred.size(0))

            loss.backward()
            optimizer.step()
            # train_loss.append(loss.item())  # 将loss这个tensor转化为标量加到列表
            batch_time.update(time.time() - end)
            end = time.time()
            # if i % 1 == 0:
            print('Time: {time}\n Epoch: [{0}/{1}]\tTrain: [{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                e,epoch,i, len(train_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                top1=top1, time=time.asctime(time.localtime(time.time()))))
            # print("Epoch %d/%d| Step %d/%d| Loss: %.2f" % (e, epoch, i, len(train_dataset) // batchsize, loss))
            # yl = yl + loss
            # if is_vis and (i + 1) % 100 == 0:
            #     vis.line(np.array([yl.cpu().item() / (i + 1)]), np.array([i + e * len(train_dataset) // batchsize]),
            #              win=viswin1, update='append')



            if i == len(train_loader) - 1:
                val_dataset2 = RandomDataset2(transform=transforms.Compose([
                    transforms.Resize([imageRandomCrop, imageRandomCrop]),
                    # transforms.CenterCrop([448, 448]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    )]))

                val_loader2 = torch.utils.data.DataLoader(
                    val_dataset2, batch_size=batchsize, shuffle=True,
                    num_workers=4, pin_memory=True)

                prec1,test_Loss = val2(val_loader2, net, criterion,e,epoch,namePostfix)
                test_Loss_list2.append(test_Loss)
                test_Accuracy_list2.append(prec1)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': e + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': best_prec1,
                    # 'optimizer_conv': optimizer_conv.state_dict(),
                    # 'optimizer_fc': optimizer_fc.state_dict(),
                }, is_best,namePostfix)

                val_dataset3 = RandomDataset3(transform=transforms.Compose([
                    transforms.Resize([imageRandomCrop, imageRandomCrop]),
                    # transforms.CenterCrop([448, 448]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    )]))

                val_loader3 = torch.utils.data.DataLoader(
                    val_dataset3, batch_size=batchsize, shuffle=True,
                    num_workers=4, pin_memory=True)

                prec13, test_Loss3 = val3(val_loader3, net, criterion, e, epoch,namePostfix)
                test_Loss_list3.append(test_Loss3)
                test_Accuracy_list3.append(prec13)

        # f1.close()
        train_Loss_list.append(softmax_losses.avg)
        train_Accuracy_list.append(top1.avg)

        # if (e + 1) % 10 == 0:

        polt_curve(e+1, train_Accuracy_list, train_Loss_list, test_Accuracy_list2, test_Loss_list2,
                   test_Accuracy_list3, test_Loss_list3,namePostfix)
        result = open('./models_pkl/result2_2_3_'+namePostfix+'.txt', 'w')
        for index in range(e+1):
            result.write(
                '{}|{}|{}|{}|{}|{}|{}\n'.format(index + 1, train_Accuracy_list[index], train_Loss_list[index],
                                                test_Accuracy_list2[index], test_Loss_list2[index],
                                                test_Accuracy_list3[index], test_Loss_list3[index]))
        result.close()
            # torch.save(net, "./models_pkl/ResNeSt50_epoch" + str(e + 1) + ".pkl")



def val2(val_loader, net, criterion,e,epoch,namePostfix):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()

    # top5 = AverageMeter()
    #
    # # switch to evaluate mode
    # model.eval()
    end = time.time()

    # net = ResNeSt().to(device)
    # f1 = open('F:\SoftwareProgram\Data\dataset\snow leopard\original\\test_negative.txt', 'w')
    # f1 = open('./test_negative2_2_'+namePostfix+'.txt', 'w')
    net.eval()
    with torch.no_grad():
        for i, (inputs, targets,image_paths) in enumerate(val_loader):

            inputs = inputs.cuda()
            targets = targets.long().cuda().squeeze()  # 对变量维度进行压缩

            # compute output
            logits = net(inputs)
            loss = criterion(logits, targets)

            #logits.topk(6, 1, True, True)

            prec1,correct= accuracy(logits, targets, 1)
            # for index,prec_index in enumerate(correct):
            #     if prec_index == False:
            #         f1.write('{},,,{}\n'.format(image_paths[index],targets[index]))   #输出预测错误的图像到test_negative.txt
            # prec5 = accuracy(logits, targets, 5)
            softmax_losses.update(loss.item(), logits.size(0))
            top1.update(prec1, logits.size(0))
            # top5.update(prec5, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % 1 == 0:
                print('Time: {time}\nEpoch: [{0}/{1}]Test: [{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        e,epoch,i, len(val_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                        top1=top1, time=time.asctime(time.localtime(time.time()))))

        # test_Loss_list.append(softmax_losses.avg)
        # test_Accuracy_list.append(100 * top1.avg)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # f1.close()

    return top1.avg,softmax_losses.avg

def val3(val_loader, net, criterion,e,epoch,namePostfix):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()

    # top5 = AverageMeter()
    #
    # # switch to evaluate mode
    # model.eval()
    end = time.time()

    # net = ResNeSt().to(device)
    # f1 = open('F:\SoftwareProgram\Data\dataset\snow leopard\original\\test_negative.txt', 'w')
    # f1 = open('./test_negative2_3_'+namePostfix+'.txt', 'w')
    net.eval()
    with torch.no_grad():
        for i, (inputs, targets,image_paths) in enumerate(val_loader):

            inputs = inputs.cuda()
            targets = targets.long().cuda().squeeze()  # 对变量维度进行压缩

            # compute output
            logits = net(inputs)
            loss = criterion(logits, targets)

            #logits.topk(6, 1, True, True)

            prec1,correct= accuracy(logits, targets, 1)
            # for index,prec_index in enumerate(correct):
            #     if prec_index == False:
            #         f1.write('{},,,{}\n'.format(image_paths[index],targets[index]))   #输出预测错误的图像到test_negative.txt
            # prec5 = accuracy(logits, targets, 5)
            softmax_losses.update(loss.item(), logits.size(0))
            top1.update(prec1, logits.size(0))
            # top5.update(prec5, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % 1 == 0:
                print('Time: {time}\nEpoch: [{0}/{1}]Test: [{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        e,epoch,i, len(val_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                        top1=top1, time=time.asctime(time.localtime(time.time()))))

        # test_Loss_list.append(softmax_losses.avg)
        # test_Accuracy_list.append(100 * top1.avg)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # f1.close()

    return top1.avg,softmax_losses.avg



def polt_curve(epoch,tarin_Accuracy_list,tarin_Loss_list,test_Accuracy_list2,test_Loss_list2,test_Accuracy_list3,test_Loss_list3,namePostfix):
    plt.clf()
    x1 = range(0, epoch)
    x2 = range(0, epoch)
    y1 = tarin_Accuracy_list
    y2 = tarin_Loss_list
    y3 = test_Accuracy_list2
    y4 = test_Loss_list2
    y5 = test_Accuracy_list3
    y6 = test_Loss_list3

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'g-',label='train2')
    plt.plot(x1, y3, 'r-',label='test2')
    plt.plot(x1, y5, 'b-', label='test3')
    plt.title('accuracy vs. epoches')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'g-',label='train2')
    plt.plot(x2, y4, 'r-',label='test2')
    plt.plot(x2, y6, 'b-', label='test3')
    plt.xlabel('loss vs. epoches')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig('./train2_2_3_'+namePostfix+'.jpg')
    # plt.show()


if __name__ == '__main__':
    main()
