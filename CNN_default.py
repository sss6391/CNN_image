# 데이터 출처: https://www.kaggle.com/puneet6060/intel-image-classification
# 데이터 개요: 14034 개의 이미지 파일
# 데이터 예측 모델: 6 종 분류
# 적용 머신러닝 모델: CNN (Convolutional Neural Network)
# 훈련 데이터셋: 14034 건
# 검증 데이터셋: 3000 건
# 시험 데이터셋: 수집데이터로서 시험셋을 확보할 수 없으므로 고려하지 않음
# 입력 데이터: 6 종류의 이미지 파일
# 은닉층: 2개
# 출력 데이터: 6개
# 사용한 활성화 함수
# - 제1 은닉층: Relu
# - 제2 은닉층: Relu
# - Output Layer: softmax
# 사용한 손실함수: categorical_crossentropy
# 사용한 Optimizer: adam
# Tensorflow 버전: 2.0.0
# 파이썬버전: 3.7.4

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from datetime import datetime

start_time = datetime.now()
# 0. 랜덤시드 고정시키기
# np.random.seed(3)
# 1. 데이터셋 불러오기
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('seg_train/seg_train', target_size=(150,150), batch_size=3, class_mode = 'categorical')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('seg_test/seg_test', target_size=(150,150), batch_size=3, class_mode = 'categorical')
# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(150,150,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 4. 모델 학습시키기
model.fit_generator(train_generator, steps_per_epoch=15, epochs=200, validation_data = test_generator, validation_steps=5)
# 5. 모델 평가하기
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


# 5. 모델 학습 과정 표시하기

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 0.5])

# acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.plot(hist.history['val_accuracy'],'g', label='val acc')
acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()



# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
end_time = datetime.now()
print(f"\n 소요 시간: {end_time-start_time}")

# 7. 결과
# accuracy: 73.33%