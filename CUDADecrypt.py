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
    overall_time = time.perf_counter()

    #Open the image
    imgFr = cv2.imread(cfg.XOR, 1)
    if imgFr is None:
        print("File does not exist!")
        raise SystemExit(0)

    # Read hash from sent file
    f = open(cfg.HASH, "r")
    srchash = int(f.read())
    f.close()

    timer[0] = time.perf_counter()
    # Inverse Fractal XOR Phase
    imgFr = cf.FracXor(imgFr, srchash)
    timer[0] = time.perf_counter() - timer[0]
    cv2.imwrite(cfg.UnXOR, imgFr)

    timer[1] = time.perf_counter()
    # Inverse MT Phase: Intra-column pixel unshuffle
    imgMT = cf.MTUnShuffle(imgFr, srchash)
    timer[1] = time.perf_counter() - timer[1]
    cv2.imwrite(cfg.UnMT, imgMT)
    imgAr = imgMT

    #Clear catmap debug files
    if cfg.DEBUG_CATMAP:
        cf.CatmapClear()
    
    timer[2] = time.perf_counter()
    # Ar Phase: Cat-map Iterations
    cv2.imwrite(cfg.OUT, imgAr)
    dim = imgAr.shape
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)

    while (cf.sha2HashImage(imgAr)!=srchash):
        cuda.memcpy_htod(gpuimgIn, imgAr_In)
        func = mod.get_function("ArCatMap")
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        cuda.memcpy_dtoh(imgAr_In, gpuimgOut)
        imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
    timer[2] = time.perf_counter() - timer[2]
    cv2.imwrite(cfg.OUT, imgAr)
    overall_time = time.perf_counter() - overall_time

    # Print timing statistics
    print("Fractal XOR Inversion completed in " + str(timer[0]) +"s")
    print("MT Unshuffle completed in " + str(timer[1]) +"s")
    print("Arnold UnMapping completed in " + str(timer[2]) +"s")
    print("Decryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()