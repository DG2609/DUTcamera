import faiss
import pandas as pd
import numpy as np
from sklearn import preprocessing

def main():
	# Path toi file csv chua embed vector
	csv_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/embedded_faces2.csv'
	df = pd.read_csv(csv_path)
	vector_arr = []
	# Lay label
	label, vector = df['Name'].values, df['Vector'].values
	'''
	unique_label = np.unique(label)
	for name in unique_label:
		vector_temp = df.loc[df['Name']==name]
		vector_arr.append(vector_temp)
	temp = []
	X=[]
	for person in vector_arr:
		for i in range(len(person.index)):
			vec = person.iloc[i].values[2][1:-1]
			vec = [float(x) for x in vec.split(',')] # Convert to float
			temp.append(vec)
		avg_vector = [sum(sub_list) / len(sub_list) for sub_list in zip(*temp)]
		X.append(avg_vector)
	X = np.asarray(X).astype(np.float32)
	# Build FAISS index
	vector_dim = len(X[0])
	index = faiss.IndexFlatL2(vector_dim)
	faiss.normalize_L2(X)
	index.train(X)
	index.add(X)
	return index, unique_label
	'''
	# Lay vector
	X=[]
	for vec in vector:
		vec = vec[1:-1]
		vec = [float(x) for x in vec.split(',')] # Convert to float
		X.append(vec)
	X = np.asarray(X).astype(np.float32)
	# Build FAISS index
	vector_dim = len(X[0])
	index = faiss.IndexFlatL2(vector_dim)	
	faiss.normalize_L2(X)
	index.train(X)
	index.add(X)
	return index, label

if __name__ == '__main__':
	main()
