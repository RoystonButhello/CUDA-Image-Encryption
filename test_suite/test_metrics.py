import cv2
import numpy as np
import math
import time
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

#Returns the average correlation of an image in horizontal direction
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

#Returns the average correlation of an image in the vertical direction
def correlationVert(img,N):
	corr_v_arr=np.zeros([N*N])
	#print("\n length of corr_h_arr="+str(len(corr_h_arr)))
	l=0
	for i in range(0,N-1):
		for j in range(0,N-1):
			corr_v_arr[l]=correlation(img[j,i],img[j+1,i+1])
			l=l+1
			avg_corr_v=np.sum(corr_v_arr)/(len(corr_v_arr))
	return avg_corr_v

#Returns the average absolute difference in pixel color levels between plain and encrypted images
#Still Testing MAE
def MeanAbsoluteErr(img_plain,img_encrypted,M,N):
	l=0
	difference=np.zeros([M*N])
	avg_mae=0
	for i in range(0,M):
		for j in range(0,N):	
			difference[l]=(1/(M*N*3))*(abs(np.sum(img_plain[i,j])-np.sum(img_encrypted[i,j])))
			l=l+1			
	print("\nmae=\n")
	print(difference)
	avg_mae=np.sum(difference)/(len(difference))
	return avg_mae			


img_pln=cv2.imread("raytracer480.png",1)

img_rgb_pln = cv2.cvtColor(img_pln, cv2.COLOR_BGR2RGB)
img_rgb_pln=cv2.resize(img_rgb_pln,(480,480))
dim_img_pln=img_rgb_pln.shape 
N_img_pln=dim_img_pln[0]

img_enc=cv2.imread("8output.png",1)
img_rgb_enc=cv2.cvtColor(img_enc,cv2.COLOR_BGR2RGB)
img_rgb_enc=cv2.resize(img_rgb_enc,(480,480))
dim_img_rgb_enc=img_rgb_enc.shape
N_img_enc=dim_img_rgb_enc[0]

time_array=np.zeros([3])

st_1=time.time()
avg_corr_h=correlationHoriz(img_rgb_pln,N_img_pln)
time_array[0]=time.time()-st_1

#st_2=time.time()
#avg_corr_v=correlationVert(img_rgb_pln,N_img_pln)
#time_array[1]=time.time()-st_2

print("\ntime for h correlation="+str(time_array[0]))
#print("\ntime for v correlation="+str(time_array[1]))
#print(im_rgb[9,9])

print("\navg_corr_h="+str(avg_corr_h))
#print("\navg_corr_v="+str(avg_corr_v))

st_3=time.time()
avg_mae=MeanAbsoluteErr(img_rgb_pln,img_rgb_enc,N_img_pln,N_img_enc)
time_array[2]=time.time()-st_3
print("\navg_mae=\n")
print(avg_mae)
print("\ntime for mae="+str(time_array[2]))