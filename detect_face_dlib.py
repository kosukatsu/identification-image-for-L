import dlib
import numpy as np
import cv2
from imutils import face_utils 
 
import utils

class FaceDetectorDlib:
    def __init__(self, model = "./models/shape_predictor_68_face_landmarks.dat",expansion=1.5):
        self.predictor = dlib.shape_predictor(model)
        self.expansion = expansion

    def detect_face(self,img):
        image_size_y = img.shape[0]
        image_size_x = img.shape[1]

        # detect face
        detector=dlib.get_frontal_face_detector()
        face = detector(img,1)[0]
        self.face = [   face.left(),
                        face.top(),
                        face.right() - face.left(),
                        face.bottom() - face.top()]
        self.face = np.array(self.face)
        size = np.array([image_size_x, image_size_y, image_size_x, image_size_y])
        self.face = utils.cvt_absolute2relative(self.face, size, True)

        # predict landmark
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        landmark = self.predictor(img_gray, face)
        self.landmark = face_utils.shape_to_np(landmark)


        size = np.array([image_size_x, image_size_y])
        size = size.reshape(1,-1).repeat(68,axis=0)
        self.landmark = utils.cvt_absolute2relative(self.landmark,size)
    
    def calc_range(self,img,aspect_ratio):
        expansion = self.expansion

        image_size_x = img.shape[1]
        image_size_y = img.shape[0]
        size = np.array([image_size_x, image_size_y])
        size = size.reshape(1,-1).repeat(68,axis=0)
        
        landmark = utils.cvt_relative2absolute(self.landmark,size)

        center_y=(landmark[28,1] + landmark[29,1])/2.0
        for i in [27,28,29,30,33]:
            print(landmark[i])
        center_x=np.mean(landmark[[27,28,29,30,33],0])

        roi_size_x = max(landmark[16,0] - center_x, center_x - landmark[0,0])
        left    = center_x - roi_size_x * expansion
        right   = center_x + roi_size_x * expansion

        # eyebrow_y = (landmark[19,1] + landmark[24,1]) / 2.0
        # dis_nose2brow = center_y - eyebrow_y
        # top = center_y - dis_nose2brow * expansions[1]
        top = center_y - roi_size_x * expansion

        # dis_nose2chin = landmark[8,1] - center_y
        # bottom = center_y + dis_nose2chin * expansions[3]
        bottom = center_y + roi_size_x * expansion

        width  = right-left
        height = bottom-top
        
        if height / width < aspect_ratio:
            d = aspect_ratio * width - height
            top    = top - d * 0.5
            bottom = bottom + d * 0.5
        else:
            d = height / aspect_ratio - width
            left   = left - d * 0.5
            right  = right + d * 0.5
        
        size=np.array([image_size_x, image_size_y, image_size_x, image_size_y])
        self.range_pos = np.array([left ,top ,right - left, bottom - top])
        self.range_pos = utils.cvt_absolute2relative(self.range_pos , size)

    def draw_thumbnail(self, row_img, height_size = 100):
        height_row  = row_img.shape[0]
        width_row   = row_img.shape[1]
        height  = height_size
        width   = int(width_row * height_size / height_row )
        thumbnail   = cv2.resize(row_img, (width, height)) 

        size = np.array([width, height, width, height])

        # draw face detection result
        x, y, w, h = utils.cvt_relative2absolute(self.face,size)
        cv2.rectangle(thumbnail,(x,y),(x+w,y+h),(255,0,0),1)

        # draw face landmark prediction result
        for (x,y) in self.landmark:
            cv2.circle(thumbnail, (int(x * width), int(y*height)), 2,(0,255,0), -1)
        
        # draw range result
        x, y, w, h =utils.cvt_relative2absolute(self.range_pos, size)
        cv2.rectangle(thumbnail, (x, y), (x + w, y + h), (0,0,255),2)

        print("青枠が検出された顔です")
        print("緑点が検出された顔のランドマークです")
        print("上手く顔検出できない場合は")
        print("モデルをCascadeClassiferに変えてください")
        print("よろしければ画像作成ボタンを押してください")
        return thumbnail

if __name__=="__main__":
    img=cv2.imread("./IMG_5259.JPG")
    model="./models/shape_predictor_68_face_landmarks.dat"
    detector = FaceDetectorDlib(model)
    detector.detect_face(img)
    print(detector.face)
    print(detector.landmark.shape)