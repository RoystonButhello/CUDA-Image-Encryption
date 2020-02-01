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

#Returns the Mean Absolute err between the plain and encrypted images
def mean_absolute_err(img_p,img_e,N):
	initial_mae = np.sum(np.absolute((img_p.astype(np.uint8)) - (img_e.astype(np.uint8))))
	mae=initial_mae/(N*N)
	return mae

#Returns image with one replaced pixel in given location  
def replace_pixel(img_p,row,col,green,blue,red):
	img_p[row,col,0]=green
	img_p[row,col,1]=blue
	img_p[row,col,2]=red
	cv2.imwrite("raytracer480_c.png",img_p)

#Returns the NPCR value of 2 encrypted images whose plain images are same execept for a single pixel
def number_of_pixels_change_rate(img_e1,img_e2,N):
	count=0
	res=np.zeros([N,N,3])
	for i in range(0,N):
		for j in range(0,N):
			for k in range(0,3):
				if img_e1[i,j,k] != img_e2[i,j,k]:
				#res[i,j,k]=img_e2[i,j,k]-img_e1[i,j,k]
				#if res[i,j,k]<=0:
					count=count+1
	npcr=(count/(N*N))*100
	return npcr

			


#Loading plain image
img_in=cv2.imread("5imgfractal_alt.png",1)
img_e1=cv2.resize(img_in,(640,640))
dim=img_e1.shape 
N=dim[0]
	
#Loading Encrypted image
img_2=cv2.imread("5imgfractal_c.png",1)
img_e2=cv2.resize(img_2,(640,640))

#avg_corr_h=my_correl_horiz(img_pln,N)
#print("\navg_corr_horiz="+str(avg_corr_h))
#avg_corr_d=my_correl_diag(img_pln,N)
#print("\navg_corr_diag="+str(avg_corr_d))

#mae=mean_absolute_err(img_pln,img_enc,N)
#print("\nmae="+str(mae))

#replace_pixel(img_in,0,1,0,0,0)

npcr=number_of_pixels_change_rate(img_e1,img_e2,N)

#print("\nnpcr="+str(npcr))
#print("\n")
#print(np.corrcoef(img[i,j],img[i+1,j+1]))