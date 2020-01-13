import pandas as pd
import cv2
import numpy as np

# csv파일 읽기 및 에러라인 읽기 제외 (error_bad_lines=False)
data_frame = pd.read_csv("styles.csv", error_bad_lines=False)
# id 오름차순 정렬
sorted_data = data_frame.sort_values(by=["id"])
# 옷이 아닌 데이터 제외 선택
selected_data = sorted_data[sorted_data["masterCategory"].isin(["Apparel"]) &
                            sorted_data["subCategory"].isin(["Topwear", "Bottomwear"])]

# 카테고리 종류 확인
# drop_data = selected_data.drop_duplicates("articleType")
# print(drop_data["articleType"])

# 딥러닝용 라벨링 추가
selected_data.loc[ selected_data["subCategory"].str.contains("Topwear"), "1st_label" ] = 0
selected_data.loc[ selected_data["subCategory"].str.contains("Bottomwear"), "1st_label" ] = 1

# 라벨링 확인
check_data = selected_data[selected_data["1st_label"].isin([1]) ]
print(check_data["subCategory"])

selected_data.to_csv("selected_data.csv", index=False)


'''
# 선택된 데이터 id 읽기
id_data = selected_data["id"]
# id로 해당하는 이미지파일 배열로 변환
images = []
for ids in id_data:
    image = cv2.imread('images/' + str(ids))
    images.append(image)

image_datas = np.array(images, dtype=np.float32)
print('shape', image_datas.shape)
'''
