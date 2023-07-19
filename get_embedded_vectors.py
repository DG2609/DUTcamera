import os
import cv2
import pandas as pd
import requests, reqBody
import json
# Tao df
df = pd.DataFrame(columns=['Name', 'Vector'])
body = reqBody.body

temp_frame_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/temp_frame.jpg'
face_dir = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/captured_faces'
csv_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/embedded_faces2.csv'

for root, dirs, files in os.walk(face_dir, topdown=False):
	for file in files:
		if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):		
			name = root.split('/')[-1]
			im = cv2.imread(os.path.join(root, file))
			cv2.imwrite(temp_frame_path, im)
			requests_url = 'http://localhost:18081/extract'
			response=requests.post(requests_url, json=body).json()
			faces = response['data'][0]['faces']
			if not faces:
				continue
			embedded_vector = faces[0]['vec']
			print(file)
			if embedded_vector:
				df = pd.concat([df, pd.DataFrame({'Name':[f'{name}'], 'Vector':[f'{embedded_vector}']})], ignore_index=True)
			else:
				print(f'ERR: Cant detect face in {os.path.join(root, file)}.')
df.to_csv(csv_path)

