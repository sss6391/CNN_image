import pandas as pd
import cv2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from datetime import datetime
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

start_time = datetime.now()
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

X_trains = X_flowers[:-10]
X_tests = X_flowers[-10:]

y_trains = y_datas[:-10]
y_tests = y_datas[-10:]

num_classes = df.label.unique().argmax() + 1

# datagen.fit(X_trains)
# X_tests_gen = train_datagen.fit(X_tests)

Y_trains = keras.utils.to_categorical(y_trains, num_classes)
Y_tests = keras.utils.to_categorical(y_tests, num_classes)


print('X_trains : ', X_trains.shape)
print('X_tests : ', X_tests.shape)
print('Y_trains : ', Y_trains.shape)
print('Y_tests : ', Y_tests.shape)

# 이미지 증폭
datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   # vertical_flip=True,
                                   fill_mode='nearest')
# generator 입력 데이터 생성
test_gen = datagen.flow(X_tests, Y_tests)

# 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
# step_epoch = 100
epoch = 200
hist = model.fit_generator(datagen.flow(X_trains, Y_trains, batch_size=8),
                           steps_per_epoch=100, epochs=epoch)

# 모델 평가하기
print("-- Evaluate --")
scores = model.evaluate_generator(test_gen , steps=10)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

end_time = datetime.now()
print(f"\n 소요 시간: {end_time-start_time}")

# 모델 학습 과정 표시하기
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_accuracy'],'g', label='val acc')
# acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 모델 저장
file_name = "acc{0:0.2f}_step".format(scores[1]*100) + str(epoch)+ "_gen"+".h5"
model.save(file_name)
