import pandas as pd
import cv2
import numpy as np

# csv파일 읽기 및 에러라인 읽기 제외
data_frame = pd.read_csv("styles.csv", error_bad_lines=False)
# id 오름차순 정렬
sorted_data = data_frame.sort_values(by=["id"])
# 옷이 아닌 데이터 제외
selected_data = sorted_data[sorted_data["masterCategory"].isin(["Apparel"]) &
                            sorted_data["subCategory"].isin(["Topwear", "Bottomwear"])]

# 카테고리 종류 확인
drop_data = selected_data.drop_duplicates("articleType")
print(drop_data["articleType"])

'''
id_data = selected_data["id"]
# 이미지 읽어 배열로 변환
images = []
for ids in id_data:
    image = cv2.imread('images/' + str(ids))
    images.append(image)

image_datas = np.array(images, dtype=np.float32)
print('shape', image_datas.shape)
'''
