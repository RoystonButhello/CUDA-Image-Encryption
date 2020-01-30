import cv2
import numpy as np
import math

#module to collect metrics of different images
count=0
#Returns correlation coefficent of any vectors 
def correl(vector_x,vector_y):
	N=len(vector_x)
	i=0
	#Covariance components
	cv_x=0
	cv_y=0
	cov_x_y=0
	sum_x=0
	sum_y=0
	
	while i<N:
		#sum of vector_x
		sum_x=sum_x+vector_x[i]
		#sum of vector_y
		sum_y=sum_y+vector_y[i]
		#sum of x*y
		i=i+1
	
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

img_in =cv2.imread("8output.png",1)
im_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
im_rgb=cv2.resize(im_rgb,(10,10))
dim=img_in.shape 
N=dim[0]
l=0
corr_x_y_arr=np.zeros([N*N])
for i in range(0,9):
	for j in  range(0,9):
		l=l+1
		corr_x_y=correl(im_rgb[i,j],im_rgb[i+1,j+1])
		corr_x_y_arr[l]=corr_x_y
		print(corr_x_y)

avg_corr_xy=np.sum(corr_x_y_arr)/N*N
print("\navg_corr_x_y="+str(avg_corr_xy))

"""
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y
def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r
"""