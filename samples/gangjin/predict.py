# coding: utf-8
# filename: predict.py
# function: 模型预测, 输出预测结果


import os
import sys
import cv2
import time
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.balloon import balloon

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

file_name = '_epoch117_confidence0.50'

# 判断over_train目录是否存在
if os.path.isdir('./outputs/outputs%s' % file_name):
    print('The directory of outputs has been created')
else:
    os.mkdir('./outputs/outputs%s' % file_name)

# The list of diffcult images to identify
error_images = {'0B061732.jpg': 1, '0BF77F48.jpg': 0, '1F2EEB22.jpg': 5, '6682D48A.jpg': 2,
                '0C006B5C.jpg': -1, '07AEAA7B.jpg': 1, '57894128.jpg': 2, 'A801FAB0.jpg': 1,
                '9B3C50E8.jpg': 9, '82E41A6B.jpg': 5, '94D948B0.jpg': 2, 'AD393334.jpg': 1,
                '659E74CF.jpg': 1, '87256772.jpg': -2, 'AA5FFE47.jpg': 1, 'AD459A0A.jpg': 1,
                'ACA80058.jpg': -1, 'B552AD38.jpg': 1, 'C120F503.jpg': 3, 'C90653FC.jpg': -1,
                'CFA330C7.jpg': -1, 'D0BF9992.jpg': -1, 'D8AE9089.jpg': 3, 'D350E678.jpg': 1,
                'E9E9D064.jpg': 1, 'F262CD18.jpg': 2, 'F06217D1.jpg': 1, 'FC31AA02.jpg': -1,
                '4ED33548.jpg': 2, '2E66CA66.jpg': 4, 'E3926776.jpg': 3, 'EB689292.jpg': 2,
                '5B10659E.jpg': 1, '5C85A08C.jpg': 5, '4A07FDFE.jpg': 5, '4EB7E052.jpg': 2,
                '5EB3B701.jpg': 5, '8A1358AD.jpg': 1, '16FBF0A1.jpg': 3, '3B1A6E2B.jpg': -1,
                '18E7A17F.jpg': 1, '41D3CD1C.jpg': 0, '5A775A37.jpg': 2, '6F07463B.jpg': 0,
                '60DF6A8A.jpg': 3, '99EA2262.jpg': 3, '504D3F24.jpg': 2, 'FC2EFFF2.jpg': 2,
                '519C12BD.jpg': 2, '539DB34A.jpg': 2, '5EE05998.jpg': 3, '6CF9B11C.jpg': 3,
                '6D864939.jpg': 2, '6E003C75.jpg': 0, '5EB3B701.jpg': -2,
                }
# wrong labels
wrong_labels = {
    'FC31AA02.jpg': [326],
    '5EB3B701.jpg': [82, 193],
    '1F2EEB22.jpg': [1916],
    '3D171FE7.jpg': [860],
    '4A07FDFE.jpg': [630],
    '6CF9B11C.jpg': [168],
    '8DD7C9E8.jpg': [1909],
    '48BD440A.jpg': [1996],
    '97A3DBF0.jpg': [2220],
    '98BD6AC4.jpg': [215],
    '2724696C.jpg': [321]
}
# local path of trained weights file
# TODO update to your model path
GANGJIN_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_balloon_0117.h5")

config = balloon.BalloonConfig()


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.50
    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2560


config = InferenceConfig()
config.display()

# create model object inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# load weights trained on gangjin dataset
model.load_weights(GANGJIN_MODEL_PATH, by_name=True)

# GANGJIN dataset Class name
class_names = ['BG', 'gangjin']

# images path list generate
BALLOON_DIR = os.path.join(ROOT_DIR, "samples/balloon/dataset")
TEST_DATA_DIR = os.path.join(BALLOON_DIR, "test")
# test_path_list = [os.path.join(TEST_DATA_DIR, x) for x in os.listdir(TEST_DATA_DIR)]

submit_file = open("./submits/submit%s.csv" % file_name, "w")
start1 = time.clock()
test_path_list = [os.path.join(TEST_DATA_DIR, x) for x in error_images.keys()]
print(test_path_list)
# 图像-检测对象分数和边界框位置字典
scores_dict = {}
bboxes_dict = {}


def swap(xx, yy, a):
    """
    根据索引交换列表任意两个元素
    :param xx: 索引
    :param yy: 索引
    :param a: 列表
    :return: 新列表
    """
    print(type(a))
    temp = a[xx]
    a[xx] = a[yy]
    a[yy] = temp


for i, image_path in enumerate(test_path_list):
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    image_name = image_path.split("/")[-1]
    start = time.clock()
    results = model.detect([img], verbose=1)
    # print(type(results))
    bboxes = []
    # 检测对象边界框位置赋值
    bboxes = results[0]['rois']
    number = bboxes.shape[0]
    # 检测对象分数赋值
    scores = results[0]['scores']
    scores_dict[image_name] = scores
    # print(type(bboxes))
    new_labels = []

    for ii, box in enumerate(bboxes):
        image_la = list(box)
        swap(0, 1, box)
        swap(2, 3, box)
        new_labels.append(box)
        submit_file.write(str(image_name) + "," + str(int(box[0])) + " " + str(int(box[1])) + " " + str(
            int(box[2])) + " " + str(int(box[3])) + "\n")
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255),
                      thickness=2)

    bboxes_dict[image_name] = new_labels

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(number), (40, 40), font, 2, (255, 0, 0), 2)
    print("Predicting image: %s " % image_name)
    cv2.imwrite("./outputs/outputs%s/%s" % (file_name, image_name), img)

    r = results[0]

    if i > 196:
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    end = time.clock()
    print("predict %s takes %d second" % (image_name, (end - start)))
    print("The model have predict {} image files".format(i + 1))

end1 = time.clock()
print('Predict total %d images take %d minute' % (len(test_path_list), (end1 - start1) / 60))
all_dict = {'scores': scores_dict, 'bboxes': bboxes_dict}

# pickle保存image_scores对象
with open('./submits/object_all%s.pickle' % file_name, 'wb') as f:
    pickle.dump(all_dict, f)
    print('Save the dict of scores and bboxes success!')

# pickle保存image_scores对象
with open('./submits/object_scores%s.pickle' % file_name, 'wb') as f:
    pickle.dump(scores_dict, f)
    print('Save the dict of scores success!')

# pickle保存image_scores对象
with open('./submits/object_bboxes%s.pickle' % file_name, 'wb') as f:
    pickle.dump(bboxes_dict, f)
    print('Save the dict of bboxes success!')
