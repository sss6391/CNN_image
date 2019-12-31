#데이터 출처: https://www.kaggle.com/puneet6060/intel-image-classification

import os
from numpy import argmax
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
import pickle

# 모델 불러오기
file_list = os.listdir("./")
for file in file_list:
    if ".h5" in file:
        model = load_model(file)
        break

with open("model_class","rb") as file:
    content = pickle.load(file)
    print(content)

# 사용할 데이터 준비하기
image_dir = "image"
pre_dir = "pre"

image_list = os.listdir(image_dir)
pre_list = os.listdir(pre_dir)

# 이미지 크기 재조정
for file in image_list:
    target_image = file.rstrip(".jpg")+"_resized.jpg"
    if target_image not in pre_list:
        image = Image.open(image_dir+"/"+file)
        resize_image = image.resize((150, 150))
        resize_image.save(pre_dir+"/"+target_image)

# 이미지 불러오기
for file in pre_list:
    test_img = plt.imread('./pre/'+file)

    # 모델 사용하기
    test_img_reshaped = test_img.reshape((1, 150, 150, 3))
    result = model.predict(test_img_reshaped)
    print(file, 'is', content[argmax(result)])

# input("Press any keys to EXIT")
# {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}