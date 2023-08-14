import torch
import os
import cv2
import numpy as np
from torch.autograd import Variable as V
from Unet import UNet
# from deeplabv3plus import DeepLabV3Plus
# from FCN import FCN32s
# from FCN8s import FCN8s
# from model.deeplab import DeepLabV3Plus
from eval_segmentation import SegmentationEvaluator
from torchvision import transforms
from PIL import Image
import config as cfg
from att_map_aspp import Aspp_att_unet
from Aspp_unet import Aspp_unet
os.environ['CUDA_VISIBLE_DEVICES']=cfg.gpu



def make_nc_RGB(np_out):
    b = np_out[0, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = np_out[1, :, :]
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = np_out[5, :, :]
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 255
    FY4_nc = cv2.merge([b, g, r])
    cv2.imwrite('aa.png',FY4_nc)
    return FY4_nc.astype(np.uint8)

def make_nc_RGB_motion(np_out):
    b = np_out[1, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = np_out[2, :, :]
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = np_out[6, :, :]
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 255
    FY4_nc = cv2.merge([b, g, r])
    cv2.imwrite('aa.png',FY4_nc)
    return FY4_nc.astype(np.uint8)

def make_nc_RGB_H8(np_out):
    b = np_out[2, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = np_out[3, :, :]
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = np_out[13, :, :]
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 255
    H8_nc = cv2.merge([b, g, r])
    cv2.imwrite('aa.png',H8_nc)
    return H8_nc.astype(np.uint8)

def make_nc_RGB_H8_motion(np_out):
    b = np_out[3, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = np_out[4, :, :]
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = np_out[14, :, :]
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 255
    H8_nc = cv2.merge([b, g, r])
    cv2.imwrite('aa.png',H8_nc)
    return H8_nc.astype(np.uint8)

def test_one_image(img_path):
   

    npy = np.load(img_path)
    npy = np.transpose(npy,[2,0,1])   
    img = npy[:cfg.num_channels,:,:]
    # img = npy[1:17,:,:]

    img = np.array(img, np.float32)
    img = V(torch.Tensor(img).cuda())
    img = img.unsqueeze(0)
   

    net = Aspp_att_unet
    net = net(n_classes = cfg.num_classes,n_channels = cfg.num_channels).cuda(0).eval()
    # net = torch.nn.DataParallel(net, device_ids=[0,1])
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()),output_device =[0]).cuda().eval()
    net.load_state_dict(torch.load(cfg.weights))
    mask = net.forward(img).squeeze()# .cpu().data.numpy()  # .squeeze(1)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0


    mask = torch.argmax(mask, axis=0)
    # print('mask.shape: ', mask.shape)
    # print(torch.unique(mask))
    return mask
def fill_color(colors,mask):
    if type(mask) is not np.ndarray:
        mask = mask.cpu().numpy()
    # print('mask.shape: ', mask.shape)
    a, b = mask.shape
    img = np.zeros((a,b,3))
    for i in range(0,len(colors)):
        img[mask==i] = colors[i]
    return img
#wzq的调色盘
def _get_voc_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete


if __name__ == "__main__":
    colors = [(0,0,0),(255,255,255)]
    test_path = cfg.test_path

    save_path = cfg.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path) 
    nc_pre = save_path + 'nc+pre/'
    if not os.path.exists(nc_pre):
        os.mkdir(nc_pre) 
    # print(save_path)
    val_img_list = os.listdir(test_path)
    imagelist = filter(lambda x: x.find('png') == -1, os.listdir(test_path))
    val_img_list = list(imagelist)
    for test_img in val_img_list:
        pre = test_one_image(test_path + test_img)
        
        pre = fill_color(colors,pre)
        
        cv2.imwrite(save_path + test_img[:-4]+"_pre.png",pre)
        
        print(test_img, "finish one")

        # mask = cv2.imread(test_path+test_img[:-4]+".png")
        # mask = cv2.resize(mask,(1024,1024))

        ncnpy = np.load(test_path + test_img)
        ncnpy = np.transpose(ncnpy,[2,0,1])

        # cv2.imwrite(nc_pre+test_img[:-4]+".png",np.hstack((mask,make_nc_RGB_H8_motion(ncnpy),pre)))
        cv2.imwrite(nc_pre+test_img[:-4]+".png",np.hstack((make_nc_RGB_H8_motion(ncnpy),pre)))
       