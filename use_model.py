import os

# 0. 사용할 패키지 불러오기
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from PIL import Image

# 모델 불러오기
from keras.models import load_model
model = load_model('a93.33s1500e200.h5')

pre = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 사용할 데이터 준비하기
image_dir = "image"
pre_dir = "pre"

image_list = os.listdir(image_dir)
pre_list = os.listdir(pre_dir)

for file in image_list:
    source_image = file
    target_image = source_image.rstrip(".jpg")+"_resized.jpg"
    if target_image not in pre_list:
        image = Image.open(image_dir+"/"+file)
        resize_image = image.resize((150, 150))
        resize_image.save(pre_dir+"/"+target_image)

# pre_list = os.listdir(pre_dir)
pre_list = os.listdir("seg_pred")
pre_list = sorted(pre_list)
for file in pre_list:
    # test_num = plt.imread('./pre/'+file)
    test_num = plt.imread('./seg_pred/'+file)

    # 모델 사용하기
    test_num = test_num.reshape((1, 150, 150, 3))
    result = model.predict(test_num)
    print(file, 'is', pre[argmax(result)])

input("Press any keys to EXIT")
    # {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}