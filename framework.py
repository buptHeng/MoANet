import torch
import torch.nn as nn
from torch.autograd import Variable as V
import cv2
import numpy as np
import os
from Unet import UNet
from PIL import Image

from eval_segmentation import SegmentationEvaluator
from torchvision import transforms
import os
import config as cfg

from AttU_Net import AttU_Net
from R2U_Net import R2U_Net
from Aspp_unet import Aspp_unet
from att_map_aspp import Aspp_att_unet
from temporal_unet import tem_unet

os.environ['CUDA_VISIBLE_DEVICES']=cfg.gpu
def warmup_poly(optimizer,epoch,init_lr,max_iter,power,warmup_steps=10):
    if epoch<=warmup_steps:
        lr=init_lr*(float(epoch)/float(warmup_steps))
    else:
        lr=init_lr*(1-epoch/max_iter)**(power)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    return lr
class MyFrame():
    def __init__(self, name,net, loss, lr=3e-3, evalmode = False,num_class = 2,num_channels = 16):
        print('num_class: ', num_class)
        print('num_channels: ', num_channels)

        
        self.name = name
        self.net = net(n_classes = num_class,n_channels = num_channels).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
       

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr, weight_decay = 1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 20,eta_min = 1e-6)
        self.loss = loss()
        self.old_lr = lr
        self.evaluator = SegmentationEvaluator(num_class)
        self.tran = transforms.ToTensor()
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img) 
        loss = self.loss(self.mask,pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.data
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        
    def test_one_img_from_path(self, img_path):
        
        npy = np.load(img_path)
        npy = np.transpose(npy,[2,0,1])
        img = npy
        img = np.array(img, np.float32)
        img = V(torch.Tensor(img).cuda(0))
        img = img.unsqueeze(0)
        
        net = Aspp_att_unet
        net = net(n_classes = cfg.num_classes,n_channels = cfg.num_channels).cuda(0).eval()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()),output_device =[0]).cuda().eval()
        net.load_state_dict(torch.load(cfg.weights_name+self.name+'_new.th'))
        mask = net.forward(img).squeeze()# .cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = torch.argmax(mask, axis=0)
        return mask

    def eval(self,path):
        imagelist = filter(lambda x: x.find('png') == -1, os.listdir(path))
        imagelist = list(imagelist)
        # print(imagelist)
        for i in range(0,len(imagelist)):
            # print('imagelist[i]: ', imagelist[i])
            pre = self.test_one_img_from_path(path + imagelist[i])
        
            gt_img = cv2.imread(path + imagelist[i][:-4] +'.png', cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.resize(gt_img, (1024, 1024))
            gt_img = np.array(gt_img, np.float32)/255.0
            gt_img[gt_img>=0.5] = 1
            gt_img[gt_img<=0.5] = 0

           
            gt_img = V(torch.FloatTensor(gt_img).cuda())
            self.evaluator.add_batch(gt_img,pre)

        
