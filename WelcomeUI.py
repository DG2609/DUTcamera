import cv2, requests, io, os, pickle, faiss
import reqBody, faiss_init
import numpy as np
from sklearn import preprocessing
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import math 
from datetime import datetime
import time 

root = tk.Tk()
root.geometry("1000x900")
root['background']='#CCFFFF'
root.title("Welcome")

# Path toi temp frame
temp_frame_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data    emp_frame.jpg'
# Size cua frame
Width=1366
Height=768
#define function
def resize_image(image, new_width, new_height):
    resized_image = image.resize((new_width, new_height))
    return resized_image



# clock animation
WIDTH = 150
HEIGHT = 150
clock_frame = Frame(root, bg='#CCFFFF' )
clock_frame.pack(anchor='ne', padx=20, pady=20)
canvas = tk.Canvas(clock_frame, width=WIDTH, height=HEIGHT, bg=clock_frame['bg'], highlightthickness=0)
canvas.pack()
def update_clock():
    canvas.delete("all")
    now = time.localtime()
    hour = now.tm_hour % 12
    minute = now.tm_min
    second = now.tm_sec
 
    # Draw clock face
    canvas.create_oval(2, 2, WIDTH, HEIGHT, outline="black", width=2)
    # Draw hour numbers
    for i in range(12):
        angle = i * math.pi/6 - math.pi/2
        x = WIDTH/2 + 0.7 * WIDTH/2 * math.cos(angle)
        y = HEIGHT/2 + 0.7 * WIDTH/2 * math.sin(angle)
        if i == 0:
            canvas.create_text(x, y-10, text=str(i+12), font=("Helvetica", 12))
        else:
            canvas.create_text(x, y, text=str(i), font=("Helvetica", 12))
 
    # Draw minute lines
    for i in range(60):
        angle = i * math.pi/30 - math.pi/2
        x1 = WIDTH/2 + 0.8 * WIDTH/2 * math.cos(angle)
        y1 = HEIGHT/2 + 0.8 * HEIGHT/2 * math.sin(angle)
        x2 = WIDTH/2 + 0.9 * WIDTH/2 * math.cos(angle)
        y2 = HEIGHT/2 + 0.9 * HEIGHT/2 * math.sin(angle)
        if i % 5 == 0:
            canvas.create_line(x1, y1, x2, y2, fill="black", width=3)
        else:
            canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
 
    # Draw hour hand
    hour_angle = (hour + minute/60) * math.pi/6 - math.pi/2
    hour_x = WIDTH/2 + 0.5 * WIDTH/2 * math.cos(hour_angle)
    hour_y = HEIGHT/2 + 0.5 * HEIGHT/2 * math.sin(hour_angle)
    canvas.create_line(WIDTH/2, HEIGHT/2, hour_x, hour_y, fill="black", width=6)
 
    # Draw minute hand
    minute_angle = (minute + second/60) * math.pi/30 - math.pi/2
    minute_x = WIDTH/2 + 0.7 * WIDTH/2 * math.cos(minute_angle)
    minute_y = HEIGHT/2 + 0.7 * HEIGHT/2 * math.sin(minute_angle)
    canvas.create_line(WIDTH/2, HEIGHT/2, minute_x, minute_y, fill="black", width=4)
 
    # Draw second hand  
    second_angle = second * math.pi/30 - math.pi/2
    second_x = WIDTH/2 + 0.6 * WIDTH/2 * math.cos(second_angle)
    second_y = HEIGHT/2 + 0.6 * WIDTH/2 * math.sin(second_angle)
    canvas.create_line(WIDTH/2, HEIGHT/2, second_x, second_y, fill="red", width=2)
 
    canvas.after(1000, update_clock)
update_clock()

def date_time():
    now = datetime.now()
    date_string = now.strftime("%A, %d %B %Y")
    date_label.config(text=date_string)
    date_label.after(1000, date_time)
date_label = Label(clock_frame, font=('Lato-Black', 14), bg='#CCFFFF')
date_label.pack(pady = 10)
date_time()




body = reqBody.body
num_of_predictions = 5

faiss_index, faiss_label = faiss_init.main()
faiss_threshold = 1.4   # Distance > threshold => Unknown
faiss_prediction = []
previous_prediction = None

list_of_predictions=[]
count=0

# hàm update name
def update_name():
    global current_index
    if current_index < len(name_list):
        lb3.config(text=name_list[current_index])
        current_index += 1


    if current_index > len(name_list):
        lb3.after_cancel()  # Hủy lịch cập nhật
    lb3.after(2000, update_name)

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
    # Create parent frame
    parent_frame = tk.Frame(root, bg='#CCFFFF')
    parent_frame.pack(side='top', fill=None, expand=False)

    # Add image to parent frame
    image = Image.open("DUT.jpg")
    resized_img = resize_image(image, 110, 100)
    tk_img = ImageTk.PhotoImage(resized_img)
    image_label = Label(parent_frame, image=tk_img)
    image_label.pack()

    # Add "Smart Building" label to parent frame
    sb_label = tk.Label(parent_frame, text="Smart Building", font=("Lato-Black", 30, 'italic'), fg='blue', bg='#CCFFFF')
    sb_label.pack(pady=50)

    # Create child frame
    child_frame = Frame(parent_frame, bg='#CCFFFF')
    child_frame.pack(pady=50)

    # Add "WELCOME" label to child frame
    wc_label = tk.Label(child_frame, text="WELCOME", bg='blue', fg='yellow', font=('Lato-Black', 24), padx=200)
    wc_label.pack()

    # Add "saaaaaa" label to child frame
    lb3 = tk.Label(child_frame, text="saaaaaa", bg='orange', fg='yellow', font=('Lato-Black', 24), padx=80)
    lb3.pack()

    # Add "DUT" label to child frame
    dut_label = tk.Label(child_frame, text="DUT", bg='red', fg='yellow', font=('Lato-Black', 24), padx=60)
    dut_label.pack()

    # Initialize camera and variables
    cam = cam_init()
    prediction = None
    name_list = []

    # Loop to detect faces and make predictions
    while True:
        faces, frame, ret = get_faces(cam)
        predictions = []
        for face in faces:
            # Get bounding box coordinates
            x, y, w, h = face['bbox']

            # Get embedded vector for the face
            embedded_vector = face['vec']

            # Make a prediction for the face
            prediction = get_single_faiss_prediction(embedded_vector)
            predictions.append(prediction)

            # Add prediction label to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 0)
            thickness = 2
            frame = cv2.putText(frame, f'{prediction}', (w, h), font, font_scale, color, thickness, cv2.LINE_AA)

        if not ret:
            pass

        # Get final predictions and update name list
        final_predictions = get_final_prediction(predictions)
        print(final_predictions)
        name_list = final_predictions
        lb3.config(text=name_list)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy OpenCV windows
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
