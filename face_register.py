import cv2
import requests
import numpy as np
import os
from sklearn import preprocessing
import pickle
from datetime import datetime
from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import reqBody
from IPCamStreamFAISS import processing

parser = ArgumentParser()

parser.add_argument("-n", "--name", dest="name", metavar="str")
args = parser.parse_args()

csv_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/embedded_faces2.csv'
save_path = f'/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/captured_faces/{args.name}'
os.makedirs(save_path, exist_ok=True)

temp_frame_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/temp_frame.jpg'

Width = 1280
Height = 720

body = reqBody.body

# Init camera
def cam_init():
    IP = "192.168.0.171"
    PORT = "554"
    Username = "admin"
    Password = "dut123456"
    CameraURL = f"rtsp://{Username}:{Password}@{IP}:{PORT}/Streaming/Channels/101"
    cam = cv2.VideoCapture(CameraURL)
    return cam

def get_faces(cam):
    ret, frame = cam.read()
    if not ret:
        pass
    else:
        frame = cv2.resize(frame, (Width, Height))
        cv2.imwrite(temp_frame_path, frame)
        requests_url = 'http://localhost:18081/extract'
        response = requests.post(requests_url, json=body).json()
        faces = response['data'][0]['faces']
    return faces, frame

threshold = 0.6

def main():
    #cam = cv2.VideoCapture(0)
    cam = cam_init()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2
    count = 0
    while True:
        faces, frame = get_faces(cam)
        frame = processing(frame, mode='LAB')
        df = pd.DataFrame(columns=['Name', 'Vector'])
        for face in faces:
            embedded_vector = face['vec']
            x, y, w, h = face['bbox']
            prob = round(face['prob'], 2)
            if prob >= 0.3:
                cropped_face = frame[y:h, x:w]
                now = datetime.now()
                current_time = now.strftime("%H-%M-%S.%f")[:-3]
                cv2.imwrite(f'{save_path}/{args.name}_{current_time}.jpg', cropped_face)
                #cv2.imwrite(f'{save_path}/{args.name}_{current_time}.jpg', cropped_face)
            cv2.rectangle(frame, (x, y), (w, h), (225, 255, 0), 4)
            frame = cv2.putText(frame, f'{prob}', (w, h), font, fontScale, color, thickness, cv2.LINE_AA)
            
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()

