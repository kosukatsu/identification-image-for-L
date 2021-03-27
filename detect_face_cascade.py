from multiprocessing import Process, Array

import cv2
import numpy as np

import utils


class FaceDetectorCascade:
    def __init__(self, model="./models/haarcascade_frontalface_default.xml", expansion=[1.5, 2.0, 1.5, 2.5]):
        self.model = model
        self.expansion = expansion

    def _detect_face(self, img, arr):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(self.model)
        faces = face_cascade.detectMultiScale(img_gray)
        for i in range(4):
            arr[i] = faces[0][i]

    def detect_face(self, img):
        arr = Array('i', range(4))
        p = Process(target=self._detect_face, args=(img, arr))
        p.start()
        p.join(timeout=1)
        p.terminate()

        pos = np.empty(4)
        for i in range(4):
            pos[i] = arr[i]

        image_size_x = img.shape[1]
        image_size_y = img.shape[0]
        size = np.array([image_size_x, image_size_y,
                        image_size_x, image_size_y])

        self.face_pos = utils.cvt_absolute2relative(pos, size)

    def calc_range(self, img, aspect_ratio):
        expansion = self.expansion

        if self.face_pos is None:
            print("顔検出できませんでした")
        image_size_x = img.shape[1]
        image_size_y = img.shape[0]
        size = np.array([image_size_x, image_size_y,
                        image_size_x, image_size_y])
        face_x, face_y, face_w, face_h = utils.cvt_relative2absolute(
            self.face_pos, size)

        center_x = face_x+face_w/2
        center_y = face_y+face_h/2

        left = center_x - face_w * 0.5 * expansion[0]
        top = center_y - face_h * 0.5 * expansion[1]
        right = center_x + face_w * 0.5 * expansion[2]
        bottom = center_y + face_h * 0.5 * expansion[3]

        width = right-left
        height = bottom-top

        if height / width < aspect_ratio:
            d = aspect_ratio * width - height
            top = top - d * expansion[1]/(expansion[1]+expansion[3])
            bottom = bottom + d * expansion[3]/(expansion[1]+expansion[3])
        else:
            d = height / aspect_ratio - width
            left = left - d * expansion[0]/(expansion[0]+expansion[2])
            right = right + d * expansion[2]/(expansion[0]+expansion[2])

        self.range_pos = np.array([left, top, right - left, bottom - top])
        self.range_pos = utils.cvt_absolute2relative(self.range_pos, size)

    def draw_thumbnail(self, row_img, height_size=100):
        height_row = row_img.shape[0]
        width_row = row_img.shape[1]
        height = height_size
        width = int(width_row * height_size / height_row)
        thumbnail = cv2.resize(row_img, (width, height))

        size = np.array([width, height, width, height])

        # draw face detection result
        x, y, w, h = utils.cvt_relative2absolute(self.face_pos, size)
        cv2.rectangle(thumbnail, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw range result
        x, y, w, h = utils.cvt_relative2absolute(self.range_pos, size)
        cv2.rectangle(thumbnail, (x, y), (x + w, y + h), (255, 0, 0), 2)

        print("緑枠が検出された顔です")
        print("青枠が実際の証明写真です")
        print("緑枠・青枠が表示されなければもう一度顔検出ボタンを押してください")
        print("よろしければ画像作成ボタンを押してください")

        return thumbnail


if __name__ == "__main__":
    img = cv2.imread("./IMG_5259.JPG")
    model = "./models/haarcascade_frontalface_default.xml"
    detector = CascadeClassifier()
    pos = detector.detect_face(img, model)
