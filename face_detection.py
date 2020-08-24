from matplotlib.patches import Rectangle


global count
count = 0

#얼굴 눈 코 입 박스만 만드는 곳
def face_eye_trace(data,result_list):
    # print("얼굴추출....")
    for result in result_list :

        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')

        left_eye = result['keypoints']['left_eye'];
        right_eye = result['keypoints']['right_eye'];

        global count

        # pyplot.show()
        x = int(result['box'][0])
        y = int(result['box'][1])
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        width = int(result['box'][2])
        height = int(result['box'][3])

        img3 = data[y:y + height, x:x + width]
        # cv2.imshow('original', img3)
        # cv2.waitKey(0)  # 아무키나 누르면 지나감 안에 값이 1이면 그냥 지나가지만 키를 눌렀을때 반응함 cv2.destroyAllWindows()

        return img3

def input_image(input_detector,pixels_image):
    # print("이미지 넣었음.")
    detector = input_detector
    pixels = pixels_image

    try:
        # print("얼굴 위치 추적 중....")
        faces = detector.detect_faces(pixels)
        # print("얼굴 위치 추적 완료")
        print(len(faces))
    # display faces on the original image
        if len(faces) == 0:
            # print("얼굴 인식하지 못했어요")
            return []
        elif len(faces) == 1 :
            face_detection = face_eye_trace(pixels,faces)
        else :
            print("다수의 얼굴이 인식되어 종료했습니다.")
            return []
    except:
        print("얼굴 인식을 하지 못하였습니다.")
        return []

    return face_detection