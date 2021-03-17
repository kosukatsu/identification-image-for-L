import PySimpleGUI as sg
import cv2
import sys
from multiprocessing import Process, Array
import numpy as np

def detectFace(img,model,arr):
    face_cascade = cv2.CascadeClassifier(model)   
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    faces = face_cascade.detectMultiScale(img_gray)
    for i in range(4):
        arr[i]=faces[0][i]

def resize_for_show(img,large_size=400,isGray=False):
    if isGray:
        y,x = img.shape
    else:
        y,x,_ = img.shape
    rate = large_size/max(y,x)
    return cv2.resize(img, (int(x * rate), int(y * rate)))

def calc_range(faces,aspect_ratio):
    expansions=[ 1.5, 2.0, 1.5, 2.5]
    face_x,face_y,face_w,face_h=faces

    center_x=face_x+face_w/2
    center_y=face_y+face_h/2

    left    = center_x - face_w * 0.5 * expansions[0]
    top     = center_y - face_h * 0.5 * expansions[1]
    right   = center_x + face_w * 0.5 * expansions[2]
    bottom  = center_y + face_h * 0.5 * expansions[3]

    width  = right-left
    height = bottom-top
    
    if height / width < aspect_ratio:
        d = aspect_ratio * width - height
        top    = top - d * expansions[1]/(expansions[1]+expansions[3])
        bottom = bottom + d * expansions[3]/(expansions[1]+expansions[3])
    else:
        d = height / aspect_ratio - width
        left   = left - d * expansions[0]/(expansions[0]+expansions[2])
        right  = right + d * expansions[2]/(expansions[0]+expansions[2])

    return left, top, right - left, bottom - top

def cvt_mm2inch(mm):
    return mm / 25.4

def cvt__inch2mm(inch):
    return inch * 25.4

if __name__=="__main__":
    sg.theme('Light Gray 1')

    purposes=["パスポート","運転免許証","履歴書"]
    sizes=[[45,35],[30,24],[40,30]]
    layout = [
        [sg.Image(filename='', key='image',size=(10,10))],
        [sg.Frame("",[
            [sg.Text('正面から撮った写真ファイルと出力先を入力')],
            [sg.Text('画像'), sg.InputText(key="image_file",size=(52,5)), sg.FileBrowse("参照")],
            [sg.Text("出力画像"), sg.InputText("./output.jpg",size=(48,5),key="output_file"), sg.FileBrowse("参照")]
        ]
        )],
        [sg.Frame("",[
            [sg.Text('使用目的を選択するか画像サイズを直接入力')],
            [sg.Text("使用目的"), sg.Combo(purposes,key="selected_purpose",size=(46,5)), sg.Button("適用",key="bttn_apply_purpose")],
            [sg.Text("画像サイズ（縦x横[mm]）"),sg.InputText(size=(15,5),key="txt_size_y"),sg.Text("x"),sg.InputText(size=(15,5),key="txt_size_x")], 
        ])],
        [sg.Frame("オプション",[
            [sg.Text('モデル'), sg.InputText("./models/haarcascade_frontalface_default.xml", size=(50,5), key="model_file"), sg.FileBrowse("参照")],
            [sg.Text("解像度[dpi]"),sg.InputText("300",size=(47,5), key="dpi")],

        ])],
        [sg.Output(size=(64,4),key="output")],
        [sg.Submit(button_text='顔検出',key="detect_face"),sg.Submit(button_text="画像作成",key="output_image")]
    ]

    window = sg.Window('test', layout)
    while True:
        event, values = window.read()

        if event is None:
            print('exit')
            break

        if event == "bttn_apply_purpose":
            if values["selected_purpose"]:
                size=sizes[purposes.index(values["selected_purpose"])]
                window["txt_size_y"].Update(str(size[0]))
                window["txt_size_x"].Update(str(size[1]))
            else:
                print("目的を選んでから押してください")
        
        if event == "detect_face":
            if not all([values["image_file"], values["model_file"], values["txt_size_x"], values["txt_size_y"]]):
                print("空欄を入力してください")
                continue
            
            row_img = cv2.imread(values["image_file"])
            
            if row_img is None:
                print("画像が読み込めません")
                continue
            
            model = values["model_file"]
            id_size_x = int(values["txt_size_x"])
            id_size_y = int(values["txt_size_y"])
            image_y,image_x,_ = row_img.shape

            arr = Array('i',range(4))
            p = Process(target=detectFace,args=(row_img,model,arr))
            p.start()
            p.join(timeout=1)
            p.terminate()

            face_x,face_y,face_w,face_h=arr[0]/image_x,arr[1]/image_y,arr[2]/image_x,arr[3]/image_y
            show_img=resize_for_show(row_img, large_size=300)
            y,x,_=show_img.shape
            cv2.rectangle(show_img,(int(face_x*x),int(face_y*y)),(int((face_x+face_w)*x),int((face_y+face_h)*y)),(0,255,0),2)
            print("緑枠が検出した顔です")

            range_x, range_y, range_w, range_h = calc_range(arr, id_size_y / id_size_x)
            range_x /= image_x
            range_y /= image_y
            range_w /= image_x
            range_h /= image_y
            cv2.rectangle(show_img,(int(range_x*x),int(range_y*y)),(int((range_x+range_w)*x),int((range_y+range_h)*y)),(255,0,0),2)

            imgbytes=cv2.imencode(".png",show_img)[1].tobytes()
            window["image"].update(data=imgbytes)
            print("青枠が実際の証明写真です")
            print("緑枠・青枠が表示されなければもう一度顔検出ボタンを押してください")
            print("よろしければ画像作成ボタンを押してください")

        if event=="output_image":
            if not all([values["dpi"], values["output_file"]]):
                print("空白を入力してください")
                continue
            dpi=int(values["dpi"])
            output_file=values["output_file"]

            face_image = row_img[int(range_y * image_y):int((range_y + range_h) * image_y),
                                 int(range_x * image_x):int((range_x + range_w) * image_x)]
            id_size_x_inch  = cvt_mm2inch(id_size_x)
            id_size_y_inch  = cvt_mm2inch(id_size_y)
            margin = 1.05
            face_x = int(id_size_x_inch * dpi * margin)
            face_y = int(id_size_y_inch * dpi * margin)
            face_image = cv2.resize(face_image,(face_x,face_y))

            L_size = np.array([3.5,5.0]) # inch
            output_size = ( L_size * dpi ).astype(int)
            if output_size[0] < output_size[1]:
                isRot = True
                output_size = output_size[::-1]
            else:
                isRot = False
            output_image = np.ones((output_size[0],output_size[1],3)).astype(np.uint8)*255

            if output_size[0] / face_y - output_size[0] // face_y > 0.1:
                y_num = output_size[0] // face_y
            else:
                y_num = output_size[0] // face_y - 1
            
            if output_size[1] / face_x - output_size[1] //face_x >0.1:
                x_num = output_size[1] // face_x
            else:
                x_num = output_size[1] // face_x
            
            if y_num > 1:
                y_margin=int(face_y*0.1 / (y_num - 1))
            else:
                y_margin=0

            if x_num > 1:
                x_margin=int(face_x * 0.1 / (x_num - 1))
            else:
                x_margin=0

            for i in range(x_num):
                for j in range(y_num):
                    output_image[j * face_y + j * y_margin:(j + 1) * face_y + j * y_margin,
                                 i * face_x + i * x_margin:(i + 1) * face_x + i * x_margin] = face_image
                    
            
            output_image = np.roll(output_image, 
                                   (output_size[0] - face_y * y_num - y_margin * (y_num - 1 )) // 2 ,
                                   axis=0)
            output_image = np.roll(output_image, 
                                   (output_size[1] - face_x * x_num - x_margin * (x_num - 1) ) // 2 ,
                                   axis=1)

            if isRot:
                output_image=np.rot90(output_image)
            
            cv2.imwrite(output_file, output_image)
            print(output_file,"に出力しました")
    window.close()