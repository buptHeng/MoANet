import numpy as np
import os
import datetime
import shutil
from PIL import Image
def cal_dist(npyu,npyv):
    dist = np.zeros(npyu.shape)
    for i in range(npyu.shape[0]):
        for j in range(npyu.shape[1]):
            dist[i][j] = np.sqrt(npyu[i][j]*npyu[i][j] + npyv[i][j] * npyv[i][j])
    return dist

def get_xy(npyx,npyy):
    xy = np.zeros((npyx.shape[0],npyx.shape[1],2))
    for i in range(npyx.shape[0]):
        for j in range(npyy.shape[1]):
            xy[i][j][0] = int(npyx[i][j])
            xy[i][j][1] = int(npyy[i][j])
    return xy

ns = 2


ori = '/data2/yzh/new_motion/dataset/test/'

motion = '/data2/yzh/new_motion/save_test_motion/n_'+str(ns)+'/npy/'

savepath = '/data2/yzh/new_motion/data_motion_test/n_'+str(ns)+'/'

mo_feature_save = '/data2/yzh/new_motion/motion_feature_test/n_'+str(ns)+'/'

oris = os.listdir(ori)
oris.sort()
alls = []
for file in oris:
    if '.npy' in file:
        # if '202003' in file:
            alls.append(file[:-4])
print(alls)
files = os.listdir(motion)
files.sort()
print(files)


for i in range(0, len(files)):
    if os.path.exists(savepath + files[i] + '.npy'):
        print(savepath + files[i] + '.npy', 'exists, pass')
        continue
    try:
        xyuv = os.listdir(motion + files[i])
        xyuv.sort()
        # print('xyuv: ', xyuv)
        npyx = np.load(motion + files[i] +'/'+ xyuv[2])
        npyy = np.load(motion + files[i] +'/'+ xyuv[3])
        npyu = np.load(motion + files[i] +'/'+ xyuv[0])
        npyv = np.load(motion + files[i] +'/'+ xyuv[1])

        xy = get_xy(npyx,npyy)
        dist = cal_dist(npyu,npyv)

        
        nex = files[i]
        print(alls.index(nex), alls.index(nex)-ns)
        pre = alls[alls.index(nex)-ns]

        print('nex: ', nex, 'pre: ', pre)
        time_pre = datetime.datetime.strptime(pre, '%Y%m%d_%H%M')
        time_nex = datetime.datetime.strptime(nex, '%Y%m%d_%H%M')
        delay = str(time_nex - time_pre)
        if 'days' in delay:
            continue
        ls = delay.split(':')
        # print('pre: ', time_pre, delay)
        
        delay_time = int(ls[0])*60 + int(ls[1])
        print('nex: ',time_nex, delay_time)
        # 设置间隔时间
        if delay_time > 360:
            continue
    
        dist = dist / delay_time * 60
        
        ori_npy = np.load(ori+files[i]+'.npy')
        # print(ori_npy.shape)
        motion_np = np.zeros((ori_npy.shape[0],ori_npy.shape[1]))
        # print(np.unique(xy))
        for m in range(xy.shape[0]):
            for n in range(xy.shape[1]):
                # print(int(xy[m][n][0]), int(xy[m][n][1]))
                motion_np[int(xy[m][n][0])][int(xy[m][n][1])] = dist[m][n]
        # print(motion_np.shape)

        ### 先单独存motion feature
        np.save(mo_feature_save + files[i] + '.npy',motion_np)

        new = np.zeros((1024,1024,17))
        new[:,:,1:17] = ori_npy
        new[:,:,0] = motion_np
        # print(new.shape)
        np.save(savepath + files[i] + '.npy', new)
        shutil.copy(ori +files[i]+ '.png', savepath)
        print(i,'/',len(files), files[i], ' finish')
        print(len(os.listdir(mo_feature_save)))
    except Exception as exx:
            logfile = open('errors_202003.log', 'a')
            logfile.write(str(exx))
            logfile.write(savepath + files[i])
            logfile.close()
        