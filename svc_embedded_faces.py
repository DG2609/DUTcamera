from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import pickle

csv_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/embedded_faces2.csv'
model_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/models/svm_recog_model.pkl'
df = pd.read_csv(csv_path)
vector = df['Vector']

# Lay label
label, vector = df['Name'].values, df['Vector'].values
LE = preprocessing.LabelEncoder()
LE.fit(label)
Y = LE.transform(label)

# Lay vector
Norm = preprocessing.Normalizer(norm='l2')
X=[]
for vec in vector:
	vec = vec[1:-1]
	vec = [float(x) for x in vec.split(',')] # Convert to float
	X.append(vec)
#X = Norm.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size=0.4, random_state=0)
pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', probability=True)
model = make_pipeline(pca, svc)
'''
svc = SVC(kernel='rbf', class_weight='balanced', probability=True)
'''
model.fit(np.asarray(X_train), np.asarray(Y_train))

#y_proba = svc.predict_proba(X_test)
#predictions = svc.predict(X_test)

# Save model
with open(model_path,'wb') as f:
    pickle.dump(model,f)

""" 
for i in range(len(y_proba)):
	if max(y_proba[i]) <= 0.6:
		predictions[i]='Unknown'
"""
