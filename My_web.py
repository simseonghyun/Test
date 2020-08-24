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
        model = load_model('./static/keras_model/1layer_128_1_best(1)-SGD.h5')
        
    return render_template('upload.html')

@app.route('/file_uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['upload_image'].read()
        # # 파일 객체 혹은 파일 스트림을 가져오고, html 파일에서 넘겨지는 값의 이름을 file1으로 했기 때문에 file1임.
        # 업로드된 파일을 특정 폴더에저장하고,

        # convert string data to numpy array
        npimg = numpy.fromstring(f, dtype = numpy.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        import face_detection
        import predict
        global detector
        face_extract = face_detection.input_image(detector,img)
        print("얼굴추출 완료")
        if len(face_extract) == 0 :
            print("얼굴인식 못했음")
            return render_template('fail_back.html')
        else :
            # cv2.imshow('original', face_extract)
            # cv2.waitKey(0)
            #
            # cv2.imwrite('face_test.jpg', face_extract)
            #
            #
            # img = Image.fromarray(face_extract)
            # print("fromarray")
            # #BGR - > RGB 블루끼 없애줌
            # b, g, r = img.split()
            # img = Image.merge("RGB", (r, g, b))
            # # img.save("temp.jpeg")
            #
            # # create file-object in memory
            # file_object = io.BytesIO()
            #
            # img.save(file_object, 'JPEG')
            #
            # # move to beginning of file so `send_file()` it will read from start
            # file_object.seek(0)

            global model
            result,k= predict.prediction(face_extract,model)

            iu_percent = round(float(k[0][0]*100),3)
            suzy_percent = round(float(k[0][1])*100,3)

            # return send_file(file_object, mimetype='image/jpeg')
            if iu_percent > suzy_percent :
                return render_template('result.html',image_file = "image/result_iu.jpg", not_similler = "수지", not_similler_percent = suzy_percent, similler = "아이유", similler_percent = iu_percent  )
            else :
                return render_template('result.html', image_file="image/result_suzy.jpg", not_similler = "아이유", not_similler_percent = iu_percent, similler = "수지", similler_percent = suzy_percent )
    else:
        return render_template('fail_back.html')
if __name__ == '__main__':
    # debug를 True로 세팅하면, 해당 서버 세팅 후에 코드가 바뀌어도 문제없이 실행됨.
    app.run(host='127.0.0.1', debug = True, threaded = True)
