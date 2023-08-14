from datetime import date
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
# from time import time
import time

from torch.utils.data import dataset
from Unet import UNet
from framework import MyFrame,warmup_poly
from loss import focal_loss,ce_loss
from new_data import ImageFolder,randomShiftScaleRotate
from tqdm import tqdm

from AttU_Net import AttU_Net
import warnings
warnings.filterwarnings("ignore")
import config as cfg
from Aspp_unet import Aspp_unet
from att_map_aspp import Aspp_att_unet
from temporal_unet import tem_unet



# 1.Define path
# SHAPE = (1024,1024)
SHAPE = cfg.SHAPE
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
ROOT = cfg.train_path
val_path = cfg.val_path

# 2.Get data list (P.S.Data and Labels are in the same folder)
imagelist = filter(lambda x: x.find('png') == -1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-4], imagelist)

# 3.Define Module name
NAME = cfg.name 

# 4.Define batch size and epoch
BATCHSIZE_PER_CARD = cfg.brc
batchsize =torch.cuda.device_count() * BATCHSIZE_PER_CARD
print('batchsize: ', batchsize)
total_epoch = cfg.epoch 

# 5.Build frame
init_lr = cfg.lr
solver = MyFrame(NAME,Aspp_att_unet, ce_loss, init_lr,num_class = cfg.num_classes,num_channels = cfg.num_channels)

# 6.Build dataset
dataset = ImageFolder(trainlist, ROOT)
print(dataset)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=4,
    drop_last = True)

# 7.define log, time
best_IOU = 0
val_iou = 0
best_epoch = -1
log_file = cfg.log
if not os.path.exists(log_file):
    os.system(r"touch {}".format(log_file))

for epoch in range(1, total_epoch + 1):
    print('当前epoch: ', epoch)
    learning_rate = warmup_poly(solver.optimizer,epoch,init_lr,total_epoch + 1,0.9,warmup_steps=10)
    pbar = tqdm(data_loader,ncols = 70)
    train_epoch_loss = 0
    for img, mask in pbar:
        
        print('img: ',img.shape, type(img))
        print('mask: ', mask.shape)

        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
        # break
    train_epoch_loss /= len(pbar)
    print ('********')
    print ('epoch:',epoch)
    print ('train_loss:',train_epoch_loss)
    print ('SHAPE:',SHAPE)
    print('best_epoch',best_epoch)
    print('best_miou',best_IOU)
    print('learning_rate',learning_rate)
    if epoch<200 and epoch%20==0:
        solver.save(cfg.weights_name + NAME + '_new'+'.th')
        solver.eval(val_path)
        val_iou = solver.evaluator.Mean_Intersection_over_Union()
        print('val_miou',val_iou)
        if solver.evaluator.Mean_Intersection_over_Union() > best_IOU:
            solver.save(cfg.weights_name + NAME + '_best'+'.th')
            best_epoch = epoch
            best_IOU = solver.evaluator.Mean_Intersection_over_Union()
        solver.evaluator.reset()
        with open(log_file,'a') as f:
            f.write('epoch: ')
            f.write(str(epoch))
            f.write(' ')

            f.write('best_miou: ')
            f.write(str(best_IOU))
            f.write(' ')
    if epoch>200 and epoch%5==0: # 300epoch
        solver.save(cfg.weights_name + NAME + '_new'+'.th')
        print('train img shape: ', img.shape)

        solver.eval(val_path)
        val_iou = solver.evaluator.Mean_Intersection_over_Union()
        print('val_iou',val_iou)
        if solver.evaluator.Mean_Intersection_over_Union() > best_IOU:
            solver.save(cfg.weights_name + NAME + '_best'+'.th')

            best_epoch = epoch
            best_IOU = solver.evaluator.Mean_Intersection_over_Union()
        solver.evaluator.reset()
    with open(log_file,'a') as f:
        f.write('epoch: ')
        f.write(str(epoch))
        f.write(' ')

        f.write('train_loss: ')
        f.write(str(train_epoch_loss))
        f.write(' ')

        f.write('best_epoch: ')
        f.write(str(best_epoch))
        f.write(' ')

        f.write('best_miou: ')
        f.write(str(best_IOU))
        f.write(' ')

        f.write('learning_rate: ')
        f.write(str(learning_rate))
        f.write(' ')

        f.write('epoch_iou: ')
        f.write(str(val_iou))
        f.write(' ')

        strtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        f.write('time: ')
        f.write(str(strtime))
        f.write('\n')
print(best_IOU) 


