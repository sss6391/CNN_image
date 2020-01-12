import os
import random
import keras

file_list = os.listdir("./")
for file in file_list:
    if "acc80" in file:
        model = keras.models.load_model(file)
        print(file+" is loaded\n")
        break
# 모델 평가
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('seg_test/seg_test', target_size=(150,150), batch_size=3, class_mode = 'categorical')
# test_generator = test_datagen.flow_from_directory('seg_train/seg_train', target_size=(150,150), batch_size=3, class_mode = 'categorical')
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
