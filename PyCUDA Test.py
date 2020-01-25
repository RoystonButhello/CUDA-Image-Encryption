'''Basic PyCUDA Program for test purposes'''
import cv2                  #OpenCV
import os                   #Path setting and file-retrieval
import glob                 #File counting
import random               #Obviously neccessary
import numpy as np          #See above
import CONFIG               #Module with Debug flags and other constants
import time                 #Literally just timing
import hashlib              #For SHA256
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

os.chdir(CONFIG.PATH)

mod = SourceModule("""
    #include <stdint.h>
    __global__ void invert(uint8_t *in)
    {
        int block = (gridDim.x*gridDim.y)*blockIdx.z + (gridDim.x)*blockIdx.y + blockIdx.x;
        int idx = (blockDim.x*blockDim.y*blockDim.z)*block + (blockDim.x*blockDim.y)*threadIdx.z + (blockIdx.x)*threadIdx.y + threadIdx.x;
        in[idx] = 255 - in[idx];
    }
  """)

filename = "minray.png"
img = cv2.imread(filename, 1)
dim = img.shape
arrimg = np.asarray(img).reshape(-1)

gpuimg = cuda.mem_alloc(arrimg.nbytes)
cuda.memcpy_htod(gpuimg, arrimg)

func = mod.get_function("invert")
func(gpuimg, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
cuda.memcpy_dtoh(arrimg, gpuimg)

img2 = (np.reshape(arrimg,dim)).astype(np.uint8)
cv2.imshow('Reshaped Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()