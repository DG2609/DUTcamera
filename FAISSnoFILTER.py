import cv2, requests, io, os, pickle, faiss
import reqBody, faiss_init
import numpy as np
from sklearn import preprocessing

# Path toi temp frame
temp_frame_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/temp_frame.jpg'
# Size cua frame
Width=1366
Height=768
# Request body
body = reqBody.body
# So luong prediction truoc khi dua ra prediction cuoi cung
num_of_predictions = 5
# Index, label va nguong cua FAISS search
faiss_index, faiss_label = faiss_init.main()
faiss_threshold = 1.4   # Distance > threshold => Unknown
faiss_prediction = []
previous_prediction = None


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
        faces=[], frame=[]
        return faces, frame
    else:
        frame = cv2.resize(frame, (Width, Height))    
        cv2.imwrite(temp_frame_path, frame)
        requests_url = 'http://localhost:18081/extract'
        response=requests.post(requests_url, json=body).json()
        faces = response['data'][0]['faces']
        return faces, frame, ret

# Prediction bang FAISS
def get_single_faiss_prediction(vec):
    # Lay vector
    X = []
    for x in vec:
        X.append(float(x)) # Convert qua float
    # Tao search vector
    X = np.asarray(X).astype(np.float32)
    search_vector = np.array([X])
    faiss.normalize_L2(search_vector)
    # Search
    distances, ann = faiss_index.search(search_vector,k=faiss_index.ntotal)
    min_dist = distances[0][0]
    avg_dist = sum(distances[0])/len(distances[0])
    if (avg_dist-min_dist)<0.15*avg_dist:
        return 'Unknown'
    '''
    # Neu khoang cach nho nhat van lon hon nguong thi prediction la unknown
    if distances[0][0] > faiss_threshold:
        return 'Unknown'
    '''
    #print("Min distance: ", distances[0][0])
    #print("Label: ", faiss_label[ann[0][0]])
    # Lay 5 kq co distance nho nhat
    predictions = faiss_label[ann[0][0:num_of_predictions]]  # => Yeu cau luc train moi label can toi thieu 5 anh
    # Chon label xuat hien nhieu nhat trong top 5 kq
    unique, counts = np.unique(predictions, return_counts=True)
    prediction = unique[np.argmax(counts)]
    #print(distances[0][0])
    #print(prediction)
    return prediction

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
            if max(counts) < 0.5*len(list_of_predictions):
                final_predictions.append('Unknown')
            else:
                list_of_predictions = [x for x in list_of_predictions if x != prediction]
                final_predictions.append(prediction)
        count=0
        list_of_predictions=[]
    return final_predictions
   
def main():
    #cam = cv2.VideoCapture(0)
    cam = cam_init()
    count=0
    prediction = None
    while(1):
        faces, frame, ret = get_faces(cam)
        predictions=[]
        for face in faces:
            # Lay bbox
            x,y,w,h = face['bbox']
            # Ve bbox
            cv2.rectangle(frame, (x,y), (w, h), (0,0,0), 4)
            # Lay embedded vector tuong ung voi khuong mat
            embedded_vector=face['vec']
            prediction = get_single_faiss_prediction(embedded_vector)
            predictions.append(prediction)
            #prediction = get_final_prediction(get_single_faiss_prediction(embedded_vector))
            
            # Ve prediction
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1  
            color = (0, 0, 0)
            thickness = 2
            frame = cv2.putText(frame, f'{prediction}', (w,h) , font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        if not ret:
            pass
        cv2.imshow('Frame', frame)
        final_predictions = get_final_prediction(predictions)
        print(final_predictions)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
