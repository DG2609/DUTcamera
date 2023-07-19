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
faiss_threshold = 1.4       # Distance > threshold => Unknown
faiss_prediction = []
previous_prediction = None
# Anti glare
clahe_h = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
clahe_s = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
clahe_v = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
clahe_l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))

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

# Process frame
def processing(frame, mode='LAB'):
    if (mode == 'HSV'):    # HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        h = clahe_h.apply(h)
        s = clahe_s.apply(s)
        v = clahe_v.apply(v)
        processed_frame = cv2.merge([h, s, v])
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_HSV2BGR)
    elif (mode == 'LAB'):    # LAB color space
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_frame)
        l = clahe_l.apply(l)
        processed_frame = cv2.merge([l, a, b])
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_LAB2BGR)
    return processed_frame
        
# Lay list cac khuong mat
def get_faces(cam):
    ret, frame = cam.read()
    if not ret:
        pass
    else:
        frame = cv2.resize(frame, (Width, Height))    
        # Lua chon giua color space LAB vs HSV. Qua testing thay LAB co ve tot hon  
        #processed_frame = frame
        processed_frame = processing(frame, mode='LAB')
        cv2.imwrite(temp_frame_path, processed_frame)
        requests_url = 'http://localhost:18081/extract'
        response=requests.post(requests_url, json=body).json()
        faces = response['data'][0]['faces']
        return faces, processed_frame, ret

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
    print(distances[0][0])
    print(prediction)
    return prediction

"""
def get_final_prediction(prediction):
    global faiss_prediction, previous_prediction
    faiss_prediction.append(prediction)
    if len(faiss_prediction) == num_of_predictions:
        unique, counts = np.unique(faiss_prediction, return_counts=True)
        final_prediction = unique[np.argmax(counts)]
        previous_prediction = final_prediction
        faiss_prediction = []
        return final_prediction
    return previous_prediction
"""   
def main():
    #cam = cv2.VideoCapture(0)
    cam = cam_init()
    count=0
    prediction = None
    while(1):
        faces, frame, ret = get_faces(cam)
        for face in faces:
            # Lay bbox
            x,y,w,h = face['bbox']
            # Ve bbox
            cv2.rectangle(frame, (x,y), (w, h), (0,0,0), 4)
            # Lay embedded vector tuong ung voi khuong mat
            embedded_vector=face['vec']
            if count == 0:
                prediction = get_single_faiss_prediction(embedded_vector)
                count=10
            else:
                count-=1
                prediction=prediction
                pass
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
