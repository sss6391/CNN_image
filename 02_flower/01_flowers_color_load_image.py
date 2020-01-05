import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

df = pd.read_csv('flower_labels.csv')
print(df.label.unique())

# 라벨 길이(분류 갯수) 0 ~ 9
df.label.unique().argmax() + 1

# filenames = df.values[:, :-1]
# filenames = filenames.reshape(-1)
filenames = df["file"] # 위 두 코드를 한줄로 표현가능
y_datas = df.values[:, -1:]

# image = cv2.imread('flower_images/' + filenames[0])
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')

flower_images = []

for filename in filenames:
    image = cv2.imread('flower_images/' + filename)
    flower_images.append(image)

X_flowers = np.array(flower_images, dtype=np.float32)
print('shape', X_flowers.shape)

# 이미지 분류
X_trains = X_flowers[:-10]
X_tests = X_flowers[-10:]

# 라벨 분류
y_trains = y_datas[:-10]
y_tests = y_datas[-10:]

num_classes = 10

Y_trains = keras.utils.to_categorical(y_trains, num_classes)
Y_tests = keras.utils.to_categorical(y_tests, num_classes)

print('X_trains : ', X_trains.shape)
print('X_tests : ', X_tests.shape)
print('Y_trains : ', Y_trains.shape)
print('Y_tests : ', Y_tests.shape)