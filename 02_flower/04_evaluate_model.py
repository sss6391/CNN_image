import os
import keras
import pandas as pd
import cv2
import numpy as np

total = 0

file_list = os.listdir("./")
for file in file_list:
    if "_gen.h5" in file:
        model = keras.models.load_model(file)
        print(file+" is loaded\n")
        break

# 라벨 불러오기
df = pd.read_csv('flower_labels.csv')
print(df.label.unique())

# filenames = df.values[:, :-1]
# filenames = filenames.reshape(-1)
filenames = df["file"] # 위 두 코드를 한줄로 표현가능
y_datas = df.values[:, -1:]

# 이미지 불러오기
flower_images = []

for filename in filenames:
    image = cv2.imread('flower_images/' + filename)
    flower_images.append(image)
X_flowers = np.array(flower_images, dtype=np.float32)
print('shape', X_flowers.shape)

# 라벨 길이(분류 갯수) 0 ~ 9
num_classes = df.label.unique().argmax() + 1

# 라벨 배열 생성
Y_tests = keras.utils.to_categorical(y_datas, num_classes)

ran = 100
for idex in range(ran):
    # 모델 평가
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow(X_flowers, Y_tests)
    print("-- Evaluate --")
    scores = model.evaluate_generator(test_gen, steps=5)
    print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
    total += float(scores[1] * 100)

total_mean = total / ran
print(total_mean)
