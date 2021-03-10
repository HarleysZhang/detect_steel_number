## 比赛地址
[智能盘点—钢筋数量AI识别](https://www.datafountain.cn/competitions/332/details)
## 环境依赖
```ubuntu, python3, tensorflow, keras, skimage, opencv-python, numpy, pandas, matplotlib等```
## 我的方案
#### 关于检测/分割模型选择
尝试了retinanet、faster rcnn、fpn和msak rcnn，其中mask rcnn得分0.980，从kaggle上得知使用U-Net全卷积网络进行语义分割可能效果比较好，目前还没有尝试。
#### 关于预训练模型
经过后期大佬分享，建议选用coco预训练模型。
#### 关于优化器选择
+ 前期选择默认SGD优化器，后来在60epoch后选择用Adam优化器。
+ I found that the model reaches a local minima faster when trained using Adam optimizer compared to default SGD optimizer。
#### 关于学习率策略
每隔25epoch，学习率下降10倍比较好。
#### 关于训练策略
Train in 3 stages: on 512x512 crops containing ships, then finetune on 1024x1024, and finally on 2048x2048. Inference on full-sized 2000x2666 images(由于时间关系没有尝试)
#### 关于图像尺寸
图像尺寸越大越好，但是注意至少要为2^6倍数，受限于硬件条件我这里是2048*2048。
#### 关于多尺度训练
每次加载图像数据，随机选择一个图像尺寸来`read image`，这样可以让模型适应于检测目标尺寸变化较大的场景。比如图像size, 可以从这个列表中选取`[514+i*32, 1024]`, `i`表示训练`iter`。
#### 关于数据增强
我不确定数据增强是否有很大效果，下面是我的数据增强方式：
```python
augmentation = iaa.Sometimes(0.6,
                             iaa.Noop(),
                             iaa.OneOf(
                                 [
                                     iaa.Fliplr(0.5),
                                     iaa.Flipud(0.5),
                                     iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                     iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)),
                                     iaa.CoarseDropout(0.02, size_percent=0.5),
                                     # iaa.Add((-40, 40), per_channel=0.5),
                                     # iaa.WithChannels(0, iaa.Affine(rotate=(0, 45))),
                                     iaa.Multiply((0.8, 1.5)),
                                     # iaa.Superpixels(p_replace=0.1, n_segments=(16, 32))
                                 ]
                             ))
```
## 使用方法
#### 1. Clone this repository
```bash
git clone https://github.com/HarleysZhang/detect_steel_number.git
```
#### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```
#### 3. Run setup from the repository root directory
```bash
python3 setup.py install
```
#### 4.Download the data 
After download the data, put it into /path/samples/gangjin/dataset, file structure is:
```
-gangjin
  - dataset/
    - rain/
      - xxx.jpg
      ...
      - via_region_data.json
    - val/
      - xxx.jpg
      ...
      - via_region_data.json
    - test/
      - xxx.jpg
  - train_labels.csv
```
#### 5.Oversample data (**Optional**) 
```bash
cd samples/gangjin/
python3 oversample_data.py
python3 read_json.py
```
#### 6. convert the csv format to json format (**Optional**) 
```bash
python3 read_json.py
```
#### 7. train the model
```bash
python gangjin.py train --dataset=./datasets/ --weights=coco
```
#### 8. predict
```bash
python3 predict.py
```
## 模型效果
DCIC 钢筋数量识别 baseline 0.98+
## Reference
[Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow](https://github.com/matterport/Mask_RCNN)
