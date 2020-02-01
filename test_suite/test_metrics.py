import cv2
import numpy as np
import math
import time

#module to collect metrics of different images
#Returns the average horizontal auto correlation -under testing
def my_correl_horiz(img,N):
	corr_h_arr=np.zeros([(N-1)*(N-1)])
	corr_mat=np.ones([2,2])
	avg_corr_h=0
	count_eq=0
	count=0
	total_corr_h=0
	l=0
	for i in range(0,N-1):
		for j in range(0,N-1):

			corr_mat=np.corrcoef(img[i,j],img[i+1,j+1])
			if math.isnan(corr_mat[0,1]):
				#nan and adjacent pixels are eual
				if all(img[i,j])==all(img[i+1,j+1]):
					corr_h_arr[l]=1
					count_eq=count_eq+1
					l=l+1
				#nan and adjacent pixels are not equal
				else:
					count=count+1
					corr_h_arr[l]=0	
					l=l+1
			else:	
				corr_h_arr[l]=corr_mat[0,1]
				l=l+1
	total_corr_h=np.sum(corr_h_arr)
	avg_corr_h=total_corr_h/(len(corr_h_arr))
	print("\ncount_eq="+str(count_eq))
	print("\ncount="+str(count))
	print("\ntotal_corr_h="+str(total_corr_h))
	print("\navg_corr_h="+str(avg_corr_h))



#Get auto correlation of an image -under testing
def get_auto_correlation(img):
	img_nd=np.asarray(img,dtype=np.uint8)
	#Flatten the image in row-major order
	img_nd=img_nd.flatten(order="C")
	corr_x_y=np.corrcoef(img_nd)
	return corr_x_y

#Returns the Mean Absolute err between the plain and encrypted images
def mean_absolute_err(img_p,img_e,N):
	initial_mae=0
	final_mae=0
	for i in range (0,N):
		for j in range(0,N):
			#for k in range(0,3):
			initial_mae=np.absolute(img_p[i,j]-img_e[i,j])/(N*N*3)
			final_mae=final_mae+initial_mae
	avg_mae=np.sum(final_mae)/(len(final_mae))	
	print("\nfinal_mae="+str(final_mae))			
	print("\navg_mae="+str(avg_mae))
	#return avg_mae

#Returns image with one replaced pixel in given location  
def replace_pixel(img_p,row,col,green,blue,red):
	img_p[row,col,0]=green
	img_p[row,col,1]=blue
	img_p[row,col,2]=red
	cv2.imwrite("raytracer480_1_px_diff.png",img_p)

#Returns the absolute difference between the pixel values of 2 encrypted images whose plain images are the same execept for a single pixel of difference
def number_of_pixels_change_rate(img_e1,img_e2,N):
	count=0
	res=np.zeros([N,N,3])
	for i in range(0,N):
		for j in range(0,N):
			for k in range(0,3):
				if img_e1[i,j,k]!=img_e2[i,j,k]:
					count=count+1
	print("\ncount="+str(count))			
	npcr=((count/(N*N))*100)/3
	return npcr

#Returns the  absolute difference between the pixel values of 2 encrypted images whose plain images are the same except for a single pixel of difference
def unified_average_changing_intensity(img_e1,img_e2,N):
	uaci=0
	for i in range(0,N):
		for j in range(0,N):
			for k in range(0,3):
				uaci_calc=(np.absolute(img_e1[i,j,k]-img_e2[i,j,k]))/(N*N*255*3)
				uaci=uaci+uaci_calc
	uaci=(uaci*100)			
	return uaci	




img_1=cv2.imread("5imgfractal.png",1)
img_e1=cv2.resize(img_1,(1024,1024))
dim=img_e1.shape 
N=dim[0]
	
#Loading Encrypted image
img_2=cv2.imread("5imgfractal_1px_diff.png",1)
img_e2=cv2.resize(img_2,(1024,1024))

#Loading plain image
img_3=cv2.imread("raytracer480.png",1)
img_pln=cv2.resize(img_3,(1024,1024))


#mean_absolute_err(img_pln,img_e1,N)
#print("\nmae="+str(mae))

#replace_pixel(img_pln,0,2,10,33,49)

#npcr=number_of_pixels_change_rate(img_e1,img_e2,N)
#print("\nnpcr="+str(npcr))

#uaci=unified_average_changing_intensity(img_e1,img_e2,N)
#print("\nuaci="+str(uaci))
#print("\navg_uaci="+str(avg_uaci))
#print("\n")
#print(np.corrcoef(img[i,j],img[i+1,j+1]))

	