import os
import random
import keras


total = 0

file_list = os.listdir("./")
for file in file_list:
    if ".h5" in file:
        model = keras.models.load_model(file)
        print(file+" is loaded\n")
        break
ran = 1000

for idex in range(ran):
    # 모델 평가
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory('seg_test/seg_test', target_size=(150,150), batch_size=3, class_mode = 'categorical')
    # test_generator = test_datagen.flow_from_directory('seg_train/seg_train', target_size=(150,150), batch_size=3, class_mode = 'categorical')
    print("-- Evaluate --")
    scores = model.evaluate_generator(test_generator, steps=5)
    print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
    total += float(scores[1] * 100)

total_mean = total / ran
print(total_mean)
