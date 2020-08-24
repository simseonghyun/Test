from flask import Flask, render_template, request, url_for, send_file
import cv2
import numpy
from PIL import Image
import io
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

app = Flask(__name__, static_url_path='/static')

from werkzeug.utils import secure_filename


# file을 submit하는 페이지
# /upload 의 페이지로 들어와서, upload.html의 파일을 렌더링하여 보여줌
# 여기서, upload.html은 프로젝트 폴더 내의 templates 폴더에 존재해야 함(default)
global detector
global model
detector = None
model = None
@app.route('/')
def render_file():
    global detector
    global model

    if detector == None :
        detector = MTCNN()

    if model == None:
        model = load_model('static/keras_model/1layer_128_1_best(1)-SGD.h5')

    return render_template('upload.html')

@app.route('/file_uploaded', methods = ['GET', 'POST'])
def upload_file():
    return render_template('fail_back.html')
if __name__ == '__main__':
    # debug를 True로 세팅하면, 해당 서버 세팅 후에 코드가 바뀌어도 문제없이 실행됨.
    app.run(host='127.0.0.1', debug = True, threaded = True)
