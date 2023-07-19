import cv2
import requests
import io
import numpy as np
import os
from sklearn import preprocessing
import pickle
from datetime import datetime
import reqBody
temp_frame_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/temp_frame.jpg'
model_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/models/svm_recog_model.pkl'
Width=1280
Height=720

body = reqBody.body

# Init model
with open(model_path, 'rb') as f:
    svc = pickle.load(f)
Norm = preprocessing.Normalizer(norm='l2')

# Init camera
def cam_init():
    IP = "192.168.0.171"
    PORT = "554"
    Username = "admin"
    Password = "dut123456"
    CameraURL = f"rtsp://{Username}:{Password}@{IP}:{PORT}/Streaming/Channels/101"
    cam = cv2.VideoCapture(CameraURL)
    return cam
# Stream URL: rtsp://admin:dut123456@192.168.0.171:554/Streaming/Channels/101
# Lay list cac khuong mat
def get_faces(cam):
    ret, frame = cam.read()
    if not ret:
        pass
    else:
        frame = cv2.resize(frame, (Width, Height))
        cv2.imwrite(temp_frame_path, frame)
        requests_url = 'http://localhost:18081/extract'
        response=requests.post(requests_url, json=body).json()
        faces = response['data'][0]['faces']
    return faces, frame

def get_prediction(vec):
    # Lay vector
    X = []
    for x in vec:
        X.append(float(x)) # Convert to float
    #vec = Norm.transform(vec) # Normalize
    # Predict
    #X = np.reshape(X, (-1,1))
    X = [X]
    Y_proba = svc.predict_proba(X) # Prob cua predictions
    Y_prediction = svc.predict(X) # Predictions
    
    return Y_prediction
    
list_of_predictions=[]
count=0
def get_final_prediction(predictions):
    global count, list_of_predictions
    count+=1
    final_predictions=[]
    for prediction in predictions:
        list_of_predictions.append(prediction)
    if count >= 15:
        for prediction in predictions:
            unique, counts = np.unique(list_of_predictions, return_counts=True)
            prediction = unique[np.argmax(counts)]
            list_of_predictions.remove(prediction)
            final_predictions.append(prediction)
        count=0
        list_of_predictions=[]
    return final_predictions
    
def main():
    #cam = cv2.VideoCapture(0)
    cam = cam_init()
    while(1):
        faces, frame = get_faces(cam)
        predictions=[]
        for face in faces:
            # Lay bbox
            x,y,w,h = face['bbox']
            
            # Ve bbox
            cv2.rectangle(frame, (x,y), (w, h), (225,255,0), 4)
            
            # Lay embedded vector tuong ung voi khuong mat
            embedded_vector=face['vec']
            prediction = get_prediction(embedded_vector)
            predictions.append(prediction)
            
            # ve prediction
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1  
            color = (255, 255, 0)
            thickness = 2
            frame = cv2.putText(frame, f'{prediction}', (w,h) , font, 
                   fontScale, color, thickness, cv2.LINE_AA)
                 
            """  
               # Luu anh khuong mat 
            if count==14:
                now = datetime.now()
                current_time = now.strftime("%H-%M-%S")
                current_date = now.strftime("%y-%m-%d")
                captured_face_path = f'/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/captured_faces/{current_date}'
                folder_exist = os.path.isdir(captured_face_path)
                if not folder_exist:
                    os.mkdir(captured_face_path)
                    
                cropped_face=frame[y:h, x:w]
                #cv2.imshow('Frame2', cropped_face)
                cv2.imwrite(f'{captured_face_path}/captured_{current_time}.jpg', cropped_face)
                count=0
            """
        cv2.imshow('Frame', frame)
        final_predictions = get_final_prediction(predictions)
        print(final_predictions)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
