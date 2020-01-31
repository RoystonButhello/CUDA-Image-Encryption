import cv2
import numpy as np
import math
import time

#module to collect metrics of different images
#Returns the average horizontal auto correlation 
def my_correl_horiz(img,N):
	l=0
	arr_dim=N-1
	two_d_mat=np.zeros([2,2])
	corr_array=np.zeros([arr_dim*arr_dim])
	for i in range(0,N-1):
		for j in range(0,N-1):
			#if i==j:
			two_d_mat=np.corrcoef(img[i,j],img[i+1,j+1])
			if math.isnan(two_d_mat[0,1]):
				corr_array[l]=0
				l=l+1
			else:
				corr_array[l]=two_d_mat[0,1]
				l=l+1
	#print("\ncorrelation array=\n")
	#print(corr_array)
	return (np.sum(corr_array)/(len(corr_array)))

#Returns the average diagonal auto correlation 
def my_correl_diag(img,N):
	l=0
	arr_dim=N-1
	two_d_mat=np.zeros([2,2])
	corr_array=np.zeros([arr_dim*arr_dim])
	for i in range(0,N-1):
		for j in range(0,N-1):
			if i==j:
				two_d_mat=np.corrcoef(img[i,j],img[i+1,j+1])
			if math.isnan(two_d_mat[0,1]):
				corr_array[l]=0
				l=l+1
			else:
				corr_array[l]=two_d_mat[0,1]
				l=l+1
	#print("\ncorrelation array=\n")
	#print(corr_array)
	return (np.sum(corr_array)/(len(corr_array)))	

def mean_absolute_err(img_p,img_e,N):
	initial_mae = np.sum(np.absolute((img_p.astype(np.uint8)) - (img_e.astype(np.uint8))))
	mae=initial_mae/(N*N)
	print("\nmae="+str(mae))

#Loading plain image
img_in=cv2.imread("raytracer480.png",1)
img_pln=cv2.resize(img_in,(480,480))
dim_p=img_pln.shape 
N=dim_p[0]
	
#Loading Encrypted image
img_2=cv2.imread("8output.png",1)
img_enc=cv2.resize(img_2,(480,480))
#avg_corr_h=my_correl_horiz(img_pln,N)
#print("\navg_corr_horiz="+str(avg_corr_h))
#avg_corr_d=my_correl_diag(img_pln,N)
#print("\navg_corr_diag="+str(avg_corr_d))
mean_absolute_err(img_pln,img_enc,N)
#print("\n")
#print(np.corrcoef(img[i,j],img[i+1,j+1]))