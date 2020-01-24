import cv2                  #OpenCV
import os                   #Path setting and file-retrieval
import glob                 #File counting
import random               #Obviously neccessary
import numpy as np          #See above
import CONFIG               #Module with Debug flags and other constants
import time                 #Literally just timing
import hashlib              #For SHA256

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

os.chdir(CONFIG.PATH)

def sha2Hash(filename):
    hashobj = hashlib.sha256()
    with open(filename,'rb') as f:
        while True:
            block = f.read(CONFIG.BUFF_SIZE)
            if not block:
                break
            hashobj.update(block)
    return int(hashobj.hexdigest(),16)

# Mersenne-Twister Intra-Column-Shuffle
def MTUnShuffle(img_in, imghash):
    mask = 2**CONFIG.MASK_BITS - 1   # Default: 8 bits
    temphash = imghash
    dim = img_in.shape
    N = dim[0]
    img_out = img_in.copy()

    for j in range(N):
        random.seed(temphash & mask)
        MTmap = list(range(N))
        random.shuffle(MTmap)
        temphash = temphash>>CONFIG.MASK_BITS
        if temphash==0:
            temphash = imghash
        for i in range(N):
            index = int(MTmap[i])
            img_out[index][j] = img_in[i][j]
    return img_out

#XOR Image with a Fractal
def FracXor(imghash):
    #Open the image
    img_in = cv2.imread("5imgfractal.png", 1)

    #Select a file for use based on hash
    fileCount = len(glob.glob1("fractals","*.png"))
    fracID = (imghash % fileCount) + 1
    filename = "fractals\\" + str(fracID) + ".png"
    #Read the file, resize it, then XOR
    fractal = cv2.imread(filename, 1)
    dim = img_in.shape
    fractal = cv2.resize(fractal,(dim[0],dim[1]))
    img_out = cv2.bitwise_xor(img_in,fractal)

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
def Decrypt():
    filename = "5imgfractal.png"

    timer = np.zeros(4)
    overall_time = time.time()

    # Read hash from sent file
    f = open("hash.txt", "r")
    srchash = int(f.read())
    f.close()

    timer[0] = time.time()
    # Inverse Fractal XOR Phase
    imgFr = FracXor(srchash)
    timer[0] = time.time() - timer[0]
    cv2.imwrite("6imgunfractal.png", imgFr)

    timer[1] = time.time()
    # Inverse MT Phase: Intra-column pixel unshuffle
    imgMT = MTUnShuffle(imgFr, srchash)
    timer[1] = time.time() - timer[1]
    cv2.imwrite("7mtunshuffle.png", imgMT)
    imgAr = imgMT

    #Clear catmap debug files
    if CONFIG.DEBUG_CATMAP:
        files = os.listdir("catmap")
        for f in files:
            os.remove(os.path.join("catmap", f))
    
    timer[2] = time.time()
    #Write initial empty file
    cv2.imwrite("8output.png", np.zeros_like(imgAr))
    # Ar Phase: Cat-map Iterations
    dim = imgAr.shape
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    while (sha2Hash("8output.png")!=srchash):
        cuda.memcpy_htod(gpuimgIn, imgAr_In)
        func = mod.get_function("ArCatMap")
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        cuda.memcpy_dtoh(imgAr_In, gpuimgOut)
        imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
        cv2.imwrite("8output.png", imgAr)
    timer[2] = time.time() - timer[2]
    cv2.imwrite("8output.png", imgAr)

    overall_time = time.time() - overall_time
    print("Fractal XOR Inversion completed in " + str(timer[0]) +"s")
    print("MT Unshuffle completed in " + str(timer[1]) +"s")
    print("Arnold UnMapping completed in " + str(timer[2]) +"s")
    print("Decryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()