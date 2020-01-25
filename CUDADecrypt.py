import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import glob                 # File counting
import time                 # Timing Execution
import random               # Obviously neccessary
import numpy as np          # See above
import hashlib              # For SHA256
import CONFIG               # Debug flags and constants
import CoreFunctions as cf  # Common functions

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

os.chdir(CONFIG.PATH)

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
def Decrypt():
    #Initialize Timer
    timer = np.zeros(4)
    overall_time = time.time()

    # Read hash from sent file
    f = open("hash.txt", "r")
    srchash = int(f.read())
    f.close()

    timer[0] = time.time()
    # Inverse Fractal XOR Phase
    imgFr = cf.FracXor("5imgfractal.png", srchash)
    timer[0] = time.time() - timer[0]
    cv2.imwrite("6imgunfractal.png", imgFr)

    timer[1] = time.time()
    # Inverse MT Phase: Intra-column pixel unshuffle
    imgMT = cf.MTUnShuffle(imgFr, srchash)
    timer[1] = time.time() - timer[1]
    cv2.imwrite("7mtunshuffle.png", imgMT)
    imgAr = imgMT

    #Clear catmap debug files
    if CONFIG.DEBUG_CATMAP:
        cf.CatmapClear()
    
    timer[2] = time.time()
    # Ar Phase: Cat-map Iterations
    cv2.imwrite("8output.png", imgAr)
    dim = imgAr.shape
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)

    while (cf.sha2Hash("8output.png")!=srchash):
        cuda.memcpy_htod(gpuimgIn, imgAr_In)
        func = mod.get_function("ArCatMap")
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        cuda.memcpy_dtoh(imgAr_In, gpuimgOut)
        imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
        cv2.imwrite("8output.png", imgAr)
    timer[2] = time.time() - timer[2]
    cv2.imwrite("8output.png", imgAr)
    overall_time = time.time() - overall_time

    # Print timing statistics
    print("Fractal XOR Inversion completed in " + str(timer[0]) +"s")
    print("MT Unshuffle completed in " + str(timer[1]) +"s")
    print("Arnold UnMapping completed in " + str(timer[2]) +"s")
    print("Decryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()