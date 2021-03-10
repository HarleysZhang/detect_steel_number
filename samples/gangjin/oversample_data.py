# coding: utf-8
# filename: oversample_data.py
# function: 对数据进行物理过采样，同时增加标签bboxs信息


import os
import shutil
import cv2
import pandas as pd
from skimage import io

SRC_DIR = os.getcwd()    # 当前目录路径
TRAIN_DIR = os.path.join(SRC_DIR, "dataset/train")    # 训练集路径
print(TRAIN_DIR)

# 判断over_train目录是否存在
if os.path.isdir('./dataset/over_train'):
    print('The directory of new_train has been creaated')
else:
    os.mkdir('./dataset/over_train')

# overs = ['5A589275.jpg', '34AEBFA3.jpg']
# new_overs = ['5A589275_0.jpg', '34AEBFA3_1.jpg', '5A589275.jpg_0', '34AEBFA3.jpg_1']

overs = [
    '0BFB817C.jpg',
    '0EAC74AF.jpg',
    '0EDF9500.jpg',
    '1B82FF85.jpg',
    '4D3BE5D5.jpg',
    '5BE19523.jpg',
    '5C94381A.jpg',
    '8DE6C059.jpg',
    '08F7E2BB.jpg',
    '67B4B0A2.jpg',
    '92AB4085.jpg',
    '147DAD30.jpg',
    '256C1DBD.jpg',
    '296A3EA4.jpg',
    '898E0058.jpg',

    '3794BD4B.jpg',
    '8702EAB5.jpg',
    '8777E599.jpg',
    '2338148B.jpg',
    '2723542B.jpg',
    'B3DF643D.jpg',
    'B4EEFB95.jpg',
    'B9E60B56.jpg',
    'BB4402AE.jpg',
    'C06245D1.jpg',
    'C2871199.jpg',
    'D0D2EED9.jpg',
    'DEBA2239.jpg',
    'E636D4C4.jpg',
    'EDB2FA69.jpg',
    'F6E0119D.jpg'

]


train_path_list = [os.path.join(TRAIN_DIR, x) for x in overs]
num_overs = 2
new_overs = []
for i in range(num_overs):
    for ii, image_path in enumerate(train_path_list):
        # img = cv2.imread(image_path)
        # b, g, r = cv2.split(img)
        # img = cv2.merge([r, g, b])
        # img = io.imread(image_path)
        image_name = image_path.split('/')[-1].split('.')[0]
        new_name = image_name + '_' + str(i+1) + '.jpg'
        new_overs.append(new_name)
        # io.imsave("./dataset/over_train/%s" % new_name, img)
        shutil.copyfile(image_path, "./dataset/over_train/%s" % new_name)
print(new_overs)
# 判断train_labels.csv文件是否存在于当前程序所在目录
if os.path.isfile('./train_labels.csv'):
    train_df = pd.read_csv('./train_labels.csv')

    for j, image in enumerate(overs):
        aa = train_df[(train_df == image).any(1)]
        sta = j*2
        end = j*2+2

        for jj, name in enumerate(new_overs[sta:end]):
            print(new_overs[sta:end])
            bb = aa.replace(image, name)
            print(bb)
        # aa = train_df[image, ' Detection']
            train_df = pd.concat([train_df, bb], ignore_index=True)
    train_df.to_csv('./new_train_labels.csv', index=None)
else:
    print('Please put train_labels.csv to the current directory')