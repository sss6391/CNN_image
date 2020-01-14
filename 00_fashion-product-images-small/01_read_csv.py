# 데이터 출처: https://www.kaggle.com/paramaggarwal/fashion-product-images-small

import pandas as pd
import cv2
import numpy as np

# csv파일 읽기 및 에러라인 읽기 제외 (error_bad_lines=False)
data_frame = pd.read_csv("styles.csv", error_bad_lines=False)
# id 오름차순 정렬
# sorted_data = data_frame.sort_values(by=["id"])

# 옷이 아닌 데이터 제외 선택
# 사용할 데이터들 분류(갯수가 적은것, 일반적이지 않은것 제외)
# Topwear - Jackets(자켓), Shirts(셔츠), Sweaters(스웨터), Sweatshirts(맨투맨), Tops(여성 나시티), Tshirts
# bottle - Jeans, Leggings, Skirts, Shorts(반바지), Track Pants(스포츠바지), Trousers(카고바지)
selected_data = data_frame[ data_frame["masterCategory"].isin(["Apparel"]) &
                        data_frame["subCategory"].isin(["Topwear", "Bottomwear"]) &
                        data_frame["articleType"].isin(["Jackets", "Shirts", "Sweaters",
                                "Sweatshirts", "Tops", "Tshirts", "Jeans", "Leggings",
                                "Skirts", "Shorts", "Track Pants", "Trousers"]) ]


# 카테고리 종류 확인
# drop_data = selected_data.drop_duplicates("articleType")
# for a in drop_data["articleType"]:
#     print(a)

# 라벨 추가
selected_data.loc[ selected_data["subCategory"].str.contains("Topwear"), "1st_label" ] = 0
selected_data.loc[ selected_data["subCategory"].str.contains("Bottomwear"), "1st_label" ] = 1

# 라벨 확인
# check_data = selected_data[selected_data["1st_label"].isin([1]) ]
# print(check_data["usage"])

print(selected_data["baseColour"])

# 해당 이미지 출력
check_image = selected_data.loc[selected_data["articleType"].str.contains('sh'), : ]
# print(check_image["id"])

import matplotlib.pyplot as plt

fig = plt.figure()
rows = 5
cols = 5
i = 1
for ids in check_image['id']:
    file_name = 'images/' + str(ids) + '.jpg'
    img = cv2.imread(file_name)
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(str(ids) + '.jpg')
    ax.set_xticks([]), ax.set_yticks([])
    i += 1
    if i > rows*cols:
        break
plt.show()

# csv로 저장
# selected_data.to_csv("selected_styles2.csv", index=False)

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
