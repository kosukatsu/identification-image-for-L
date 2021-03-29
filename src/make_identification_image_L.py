import PySimpleGUI as sg
import cv2
from multiprocessing import Process, Array
import numpy as np

from detect_face_cascade import FaceDetectorCascade
from detect_face_dlib import FaceDetectorDlib
from utils import cvt_mm2inch, cvt_relative2absolute

purposes = ["パスポート", "運転免許証", "履歴書"]
methods = ["Dlib(Ensemble of Regression Trees)", "CascadeClassifer"]
sizes = [[45, 35], [30, 24], [40, 30]]


def spread_face(image, detector, id_size_x, id_size_y, dpi, margin=1.05):
    image_size_x = image.shape[1]
    image_size_y = image.shape[0]
    size = np.array([image_size_x, image_size_y, image_size_x, image_size_y])
    x, y, w, h = cvt_relative2absolute(detector.range_pos, size)
    face_image = image[y:y + h, x:x + w]

    id_size_x_inch = cvt_mm2inch(id_size_x)
    id_size_y_inch = cvt_mm2inch(id_size_y)
    face_x = int(id_size_x_inch * dpi * margin)
    face_y = int(id_size_y_inch * dpi * margin)
    face_image = cv2.resize(face_image, (face_x, face_y))

    L_size = np.array([3.5, 5.0])  # inch
    output_size = (L_size * dpi).astype(int)
    if output_size[0] < output_size[1]:
        isRot = True
        output_size = output_size[::-1]
    else:
        isRot = False
    output_image = np.ones(
        (output_size[0], output_size[1], 3)).astype(np.uint8)*255

    if output_size[0] / face_y - output_size[0] // face_y > 0.1:
        y_num = output_size[0] // face_y
    else:
        y_num = output_size[0] // face_y - 1

    if output_size[1] / face_x - output_size[1] // face_x > 0.1:
        x_num = output_size[1] // face_x
    else:
        x_num = output_size[1] // face_x

    if y_num > 1:
        y_margin = int(face_y*0.1 / (y_num - 1))
    else:
        y_margin = 0

    if x_num > 1:
        x_margin = int(face_x * 0.1 / (x_num - 1))
    else:
        x_margin = 0

    for i in range(x_num):
        for j in range(y_num):
            output_image[j * face_y + j * y_margin:(j + 1) * face_y + j * y_margin,
                         i * face_x + i * x_margin:(i + 1) * face_x + i * x_margin] = face_image

    output_image = np.roll(output_image,
                           (output_size[0] - face_y * y_num -
                            y_margin * (y_num - 1)) // 2,
                           axis=0)
    output_image = np.roll(output_image,
                           (output_size[1] - face_x * x_num -
                            x_margin * (x_num - 1)) // 2,
                           axis=1)

    if isRot:
        output_image = np.rot90(output_image)

    return output_image


def bttn_apply_purpose(values, window, d):
    if values["selected_purpose"]:
        size = sizes[purposes.index(values["selected_purpose"])]
        window["txt_size_y"].Update(str(size[0]))
        window["txt_size_x"].Update(str(size[1]))
    else:
        print("目的を選んでから押してください")


def bttn_detect_face(values, window, d):
    if not all([values["image_file"], values["model"], values["txt_size_x"], values["txt_size_y"], values["expansion"]]):
        print("空欄を入力してください")
        return

    row_img = cv2.imread(values["image_file"])

    if row_img is None:
        print("画像が読み込めません")
        return

    if values["model"] == methods[0]:
        detector = FaceDetectorDlib(
            expansion=float(values["expansion"])*1.5)
    elif values["model"] == methods[1]:
        detector = FaceDetectorCascade()
    else:
        print("model error")
        return

    id_size_x = int(values["txt_size_x"])
    id_size_y = int(values["txt_size_y"])

    if id_size_x <= 0 or id_size_y <= 0:
        print("画像サイズの値が不適切です")
        return

    detector.detect_face(row_img)

    detector.calc_range(row_img, id_size_y / id_size_x)

    thumbnail = detector.draw_thumbnail(row_img, height_size=150)

    imgbytes = cv2.imencode(".png", thumbnail)[1].tobytes()
    window["image"].update(data=imgbytes)

    d = {
        "row_img": row_img,
        "detector": detector,
        "id_size_x": id_size_x,
        "id_size_y": id_size_y
    }
    return d


def bttn_output_image(values, window, d):
    if not all([values["dpi"], values["output_file"]]):
        print("空白を入力してください")
        return
    if "detector" not in d:
        print("先に顔検出ボタンを押してください")

    dpi = int(values["dpi"])
    output_file = values["output_file"]

    if dpi <= 0:
        print("dpiの値が不適切です")
        return

    margin = 1.05
    margin = 1.06
    output_image = spread_face(
        d["row_img"], d["detector"], d["id_size_x"], d["id_size_y"], dpi, margin=margin)

    cv2.imwrite(output_file, output_image)
    print(output_file, "に出力しました")


def main():
    sg.theme('Light Gray 1')

    layout = [
        [sg.Image(filename='', key='image', size=(10, 10))],
        [sg.Frame("正面から撮った写真ファイルと出力先を入力", [
            [sg.Text('入力画像'), sg.InputText(key="image_file",
                                           size=(52, 5)), sg.FileBrowse("参照")],
            [sg.Text("出力画像"), sg.InputText("./output.jpg",
                                           size=(48, 5), key="output_file"), sg.FileBrowse("参照")]
        ]
        )],
        [sg.Frame("使用目的を選択するか画像サイズを直接入力", [
            [sg.Text("使用目的"), sg.Combo(purposes, default_value=purposes[2], key="selected_purpose", size=(
                46, 5)), sg.Button("適用", key="bttn_apply_purpose")],
            [sg.Text("画像サイズ（縦x横[mm]）"), sg.InputText("40", size=(15, 5), key="txt_size_y"), sg.Text(
                "x"), sg.InputText("30", size=(15, 5), key="txt_size_x")],
        ])],
        [sg.Frame("オプション", [
            [sg.Text('モデル'), sg.Combo(
                methods, default_value=methods[0], key="model")],
            [sg.Text("解像度[dpi]"), sg.InputText(
                "300", size=(47, 5), key="dpi")],
            [sg.Text("拡大率(Dlibのみ)"), sg.Slider(range=(
                0.75, 1.5), default_value=1, key="expansion", orientation="h", resolution=0.05)]
        ])],
        # [sg.Output(size=(64, 4), key="output")],
        [sg.Button(button_text='顔検出', key="bttn_detect_face"),
         sg.Button(button_text="画像作成", key="bttn_output_image")]
    ]

    handler = {
        "bttn_detect_face": bttn_detect_face,
        "bttn_output_image": bttn_output_image,
        "bttn_apply_purpose": bttn_apply_purpose,
    }

    d = {}

    window = sg.Window('test', layout)
    while True:
        event, values = window.read()

        if event is None:
            print('exit')
            break

        event_func = handler[event]
        d2 = event_func(values, window, d)
        if d2 is not None:
            d.update(d2)

    window.close()


if __name__ == "__main__":
    main()
