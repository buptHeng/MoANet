import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision import transforms
import cv2
import numpy as np
import os
from PIL import Image
import config as cfg
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if len(image.shape)!=4:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:

        image=np.rot90(image)
        #print("rotimg", image.shape)
        # print("typtofmask",type(mask))

        mask=np.rot90(mask)
        #print("rotmask", mask.shape)

    return image, mask
def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def default_loader(id, root):
    
    npy = np.load(os.path.join(root,'{}.npy').format(id))
    npy = np.transpose(npy,[2,0,1])
    img = np.array(npy, np.float32)
    #print(root+id[:-4]+".png")
    # mask = Image.open(root+id[:-4]+".png")
    # mask = mask.resize((256,256))
    # # mask = np.expand_dims(mask, axis=2)
    # img = np.array(img, np.float32)
    
    
    # mask = mask.convert("RGB")
    #print('mask: ', mask)
    mask = cv2.imread(os.path.join(root+'{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (1024, 1024))
    
    # mask = np.expand_dims(mask, axis=2)
    
    mask = np.array(mask, np.float32)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    # print('default_loader mask: ', mask.shape)
    return img, mask
    
class ImageFolder(data.Dataset):
    def __init__(self, trainlist, root):
        self.ids = list(trainlist)
        #print(self.ids)
        self.loader = default_loader
        self.root = root
        self.tran = transforms.ToTensor()
    def __getitem__(self, index):
        id = self.ids[index]

        img, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.FloatTensor(mask)

        return img, mask
    def __len__(self):
        return len(self.ids)

# from collections import  Counter
if __name__ == "__main__":
    npy = np.load("/nfs/czb/data_for_fy4a_seafog_huangbosea/20180303_0000.npy")
    print(np.unique(npy[-1,:,:]))