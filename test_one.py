import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from makeDataset import *

############## web模块 ################
from flask import request, Flask, jsonify # flask是一个使用Python编写的轻量级web应用框架
from gevent import pywsgi # gevent第三方协程库  pywsgi web服务器
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
from werkzeug.utils import secure_filename

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
## web http请求,图片识别
@app.route('/receipt', methods=['POST'])
def snowleopard():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    fileName = secure_filename(f.filename)
    # 一定要先创建该文件夹，不然会提示没有该路径
    upload_path = os.path.join(basepath, 'inputImg', fileName)
    # 保存文件
    f.save(upload_path)

    #result_info = {'result': 'ok'}
    #return jsonify(result_info)

    # time_start = time.time()
    # image_path = r'E:\cfy\myCode\xueBaoClassification\inputImg/'+ fileName
    image_path = './inputImg/'+ fileName
    result = '未知'
    cls = ['非雪豹','雪豹']
    namePostfix = 'focal_50_0.4_1000'
    # net = ResNeSt().to(device)
    net = ResNeSt()
    test_model = './models_pkl/model_best_3_3_'  + namePostfix +  '.pth.tar'
    ckpt = torch.load(test_model, map_location="cpu")
    # net.load_state_dict(ckpt["state_dict"])
    net.load_state_dict(ckpt["state_dict"])
    # print('model_epoch:',ckpt["epoch"],"test_prec:",ckpt['best_prec1'])
    net = net.cuda()
    net.eval()

    imageResize = 448
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        print(image_path)
    img = transforms.Compose([
        transforms.Resize([imageResize, imageResize]),
        # transforms.CenterCrop([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )])(img)

    inputs = img.unsqueeze(0).cuda()
    logits = net(inputs)
    _, ind = logits.topk(1)
    result = cls[ind]

    print('识别结果为: ' + result)
    # time_end = time.time()
    # print('耗时%s!!' % (time_end - time_start))
    result_info = {'result': result}
    return jsonify(result_info)

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('219.243.215.212', 8088), app)
    server.serve_forever()