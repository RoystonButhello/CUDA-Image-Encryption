import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import random               # Obviously neccessary
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

os.chdir(cfg.PATH)

# Function to equalize Luma channel
def HistEQ(img_in):
    # Convert to LAB Space
    img_lab = cv2.cvtColor(img_in, cv2.COLOR_BGR2LAB)

    # Equalize L(Luma) channel
    img_lab[:,:,0] = cv2.equalizeHist(img_lab[:,:,0])

    # Convert back to RGB Space
    img_out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    if cfg.DEBUG_HISTEQ:
        cv2.imshow('Input image', img_in)
        cv2.imshow('Histogram equalized', img_out)
    return img_out

mod = SourceModule("""
    #include <stdint.h>
    __global__ void ArCatMap(uint8_t *in, uint8_t *out)
    {
        int nx = (2*blockIdx.x + blockIdx.y) % gridDim.x;
        int ny = (blockIdx.x + blockIdx.y) % gridDim.y;
        int blocksize = blockDim.x * blockDim.y * blockDim.z;
        int InDex = ((gridDim.x)*blockIdx.y + blockIdx.x) * blocksize  + threadIdx.x;
        int OutDex = ((gridDim.x)*ny + nx) * blocksize + threadIdx.x;
        out[OutDex] = in[InDex];
    }
  """)

# Driver function
def Encrypt():
    #Initialize Timer
    timer = np.zeros(5)
    overall_time = time.perf_counter()
    
    # Open Image
    filename = cfg.IN
    img = cv2.imread(filename, 1)
    dim = img.shape

    # Check image dimensions
    timer[0] = overall_time
    if dim[0]!=dim[1]:
        N = min(dim[0], dim[1])
        img = cv2.resize(img,(N,N))
        dim = img.shape

    # Perform histogram equalization
    imgEQ = HistEQ(img)
    timer[0] = time.perf_counter() - timer[0]
    cv2.imwrite(cfg.HISTEQ, imgEQ)
    imgAr = imgEQ

    timer[1] = time.perf_counter()
    # Compute hash of imgEQ and write to text file
    imghash = cf.sha2alt(imgEq)
    timer[1] = time.time() - timer[1]
    f = open("hash.txt","w+")
    f.write(str(imghash))
    f.close()
    
    #Clear catmap debug files
    if cfg.DEBUG_CATMAP:
        cf.CatmapClear()
    
    timer[2] = time.perf_counter()
    # Ar Phase: Cat-map Iterations
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    for i in range (1, random.randint(cfg.AR_MIN_ITER,cfg.AR_MAX_ITER)):
        cuda.memcpy_htod(gpuimgIn, imgAr_In)
        func = mod.get_function("ArCatMap")
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        cuda.memcpy_dtoh(imgAr_In, gpuimgOut)
        # Write intermediate files if debugging is enabled
        if cfg.DEBUG_CATMAP:
            imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
            cv2.imwrite(cfg.ARTEMP + str(i) + ".png", imgAr)

    imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
    timer[2] = time.perf_counter() - timer[2]
    cv2.imwrite(cfg.ARMAP, imgAr)

    timer[3] = time.perf_counter()
    # MT Phase: Intra-column pixel shuffle
    imgMT = cf.MTShuffle(imgAr, imghash)
    timer[3] = time.perf_counter() - timer[3]
    cv2.imwrite(cfg.MT, imgMT)

    timer[4] = time.perf_counter()
    # Fractal XOR Phase
    imgFr = cf.FracXor(cfg.MT, imghash)
    timer[4] = time.perf_counter() - timer[4]
    cv2.imwrite(cfg.XOR, imgFr)
    overall_time = time.perf_counter() - overall_time

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Pre-processing completed in " + str(timer[0]) +"s")
        print("Hashing completed in " + str(timer[1]) +"s")
        print("Arnold Mapping completed in " + str(timer[2]) +"s")
        print("MT Shuffle completed in " + str(timer[3]) +"s")
        print("Fractal XOR completed in " + str(timer[4]) +"s")
        print("\nEncryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time\n")
    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()