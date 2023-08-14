import time
# from framework import MyFrame,warmup_poly
from AttU_Net import AttU_Net
from Unet import UNet
from loss import focal_loss,ce_loss
from R2U_Net import R2U_Net
from Aspp_unet import Aspp_unet
from att_map_aspp import Aspp_att_unet


num_classes = 2

num_channels = 17

ns = 2
name = 'n_'+str(ns)+'_Aspp_Unet'+'ns_2'+time.strftime("%Y_%m_%d", time.localtime())

train_path = '/data2/yzh/new_motion/data_motion_weight_0519/n_'+str(ns)+'/train/'

val_path = '/data2/yzh/new_motion/data_motion_weight_0519/n_'+str(ns)+'/val/'

gpu = '0, 1, 2, 3'
SHAPE = (1024,1024)
log = '/data2/yzh/new_motion/logs_weight/'+time.strftime("%Y_%m_%d", time.localtime())+'_' + name + '.log'
weights_name = '/data2/yzh/new_motion/weights_weight/'
lr = 2e-4 
brc = 1 # batch size per card
epoch = 300

# test config
test_path = '/data2/yzh/new_motion/data_motion_test/n_1/'
weights = '/data2/yzh/new_motion/weights/n1_Aspp_att_unet_third_2023_02_24_best.th'
save_path = '/data2/yzh/new_motion/save_pics_weight/n2_2/' 





