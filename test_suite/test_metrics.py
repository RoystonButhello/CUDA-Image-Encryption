import cv2
import numpy as np
import math
#module to collect metrics of different images


#Returns correlation coefficent of any vectors 
def correlation(vector_x,vector_y):
	N=len(vector_x)
	i=0
	#Covariance components
	cv_x=0
	cv_y=0
	cov_x_y=0
	sum_x=0
	sum_y=0
	
	sum_x=np.sum(vector_x)
	sum_y=np.sum(vector_y)	
	
	mean_x=sum_x/N;
	mean_y=sum_y/N;
	
	for i in range(0,N):
		cv_x=cv_x+vector_x[i]-mean_x
		cv_y=cv_y+vector_y[i]-mean_y
		#Covariance of x and y
	
	if cv_x==0 or cv_y==0:
		corr_x_y=0
		return corr_x_y

	cov_x_y=(1/N)*cv_x*cv_y
	var_x=(1/(N-1))*(cv_x*cv_x)
	var_y=(1/(N-1))*(cv_y*cv_y)
	sq_root_x=math.sqrt(var_x)
	sq_root_y=math.sqrt(var_y)
	sq_root_x_y=sq_root_x*sq_root_y
	corr_x_y=cov_x_y/sq_root_x_y
	return corr_x_y

#Returns the correlation of an image in horizontal direction
def correlationHoriz(img,N):
	corr_h_arr=np.zeros([N*N])
	#print("\n length of corr_h_arr="+str(len(corr_h_arr)))
	l=0
	for i in range(0,N-1):
		for j in range(0,N-1):
			corr_h_arr[l]=correlation(img[i,j],img[i+1,j+1])
			l=l+1
			avg_corr_h=np.sum(corr_h_arr)/(len(corr_h_arr))
	return avg_corr_h			

img_in =cv2.imread("8output.png",1)
img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
img_rgb=cv2.resize(img_rgb,(480,480))
dim=img_rgb.shape 
N=dim[0]

avg_corr_h=correlationHoriz(img_rgb,N)
#print(im_rgb[9,9])
print("\navg_corr_h="+str(avg_corr_h))
