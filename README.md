# Unet-Attention-unet-paddlepseg
基于ppseg的婴儿头围精确测量项目


## 项目目的和背景
> 超声图像中胎儿头围是评估胎儿发育（评估胎儿体重的重要参数之一）的重要指标，胎儿头围一般由医生手动进行椭圆拟合，
> 费时费力，且存在较高的用户差异，导致胎儿发育诊断的偏差，特别是会导致分娩方式决定的错误。因此基于人工智能算法帮助医生快速检测和评估胎儿头围，
> 可以在短时间内基于算法结果，协助医生选择何种治疗或者分娩方式具有重要的意义。


- 1.项目介绍
> 本项目基于paddleseg框架，采用attention-unet，实现对婴儿头围的分割，并采取图像后处理技术，对分割头像进行处理。同时根据分割后的图像进行边缘拟合从而得出婴儿的精确头围数据。

- 2.安装依赖
>  !git clone https://gitee.com/paddlepaddle/PaddleSeg.git


>  !pip install paddleseg


>  !pip install scikit-image


>  !pip install opencv-python==3.4.2.17 -i https://pypi.douban.com/simple

- 3，数据集地址
> [数据地址](https://aistudio.baidu.com/aistudio/datasetdetail/100987)

- 4.1数据处理
```
###将HC和annotation分离
import os
from glob import glob
src_path = '/home/aistudio/work/training_set'
dst_path = '/home/aistudio/work/train/label'
if os.path.exists('/home/aistudio/work/train/label'):
    print('文件夹已存在')
else:
    os.makedirs('/home/aistudio/work/train/label')
def move_file(src_path, dst_path, file):
    print('from:',src_path)
    print('to:',dst_path)
    try:
        cmd = 'chmod -R +x ' + src_path
        os.popen(cmd)
        f_src = os.path.join(src_path , "{}".format(file))
        f_dst = os.path.join(dst_path,  "{}".format(file))
        shutil.move(f_src, f_dst)
    except Exception as e:
        print ('move_file ERROR: ',e)
        traceback.print_exc()

for i in glob('/home/aistudio/work/training_set/*Annotation.png'):
    file = i.split('/')[-1]
    move_file(src_path, dst_path, file)
```
- 4.2 处理标签
```
import numpy as np
from skimage.feature import canny 
from scipy import ndimage as ndi #轮廓填充用
from skimage.io import imread, imsave
from skimage import morphology
for i in tqdm(glob('/home/aistudio/work/train/rename_label/*.png')):
    img = imread(i) # 读取的图片路径
    edges = canny(img/255.) ##canny 算子提取周围轮廓
    fill_img = ndi.binary_fill_holes(edges)  #轮廓填充
    imsave(i,fill_img)
    
    
import cv2
# from skimage import morphology # 形态学处理（后处理可能会用到）
from glob import glob
from tqdm import tqdm
#循环灰度图片并保存
def grayImg():
    for x in tqdm(glob('/home/aistudio/work/train/rename_label/*.png')):
        # print(x)
        #读取图片
        img = cv2.imread(x)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        difference = (img_gray.max() - img_gray.min()) // 2
        _, img_binary = cv2.threshold(img_gray, difference, 1, cv2.THRESH_BINARY)
        # print("阈值：", _)
        #保存灰度后的新图片
        cv2.imwrite(x,img_binary)
grayImg()

```
- 5.训练
- 6.测试
- 7.后处理
