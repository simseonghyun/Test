from cv2 import cv2
import numpy as np

# 2. 모델 불러오기
from keras.models import load_model
def prediction(img,model):
    model = model
    img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_AREA)
    x=[]
    x.append(img/256)
    img = np.array(x)

    print("예측")
    k=model.predict(img)

    for i in  k:
        if i[0] > i[1]:
            return "아이유",k
        else:
            return "수지",k
