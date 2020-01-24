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

# Function to equalize Luma channel
def HistEQ(img_in):
    # Convert to LAB Space
    img_lab = cv2.cvtColor(img_in, cv2.COLOR_BGR2LAB)

    # Equalize L(Luma) channel
    img_lab[:,:,0] = cv2.equalizeHist(img_lab[:,:,0])

    # Convert back to RGB Space
    img_out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    if CONFIG.DEBUG_HISTEQ:
        cv2.imshow('Input image', img_in)
        cv2.imshow('Histogram equalized', img_out)
    return img_out

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
def MTShuffle(img_in, imghash):
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
            img_out[i][j] = img_in[index][j]
    return img_out

#XOR Image with a Fractal
def FracXor(imghash):
    #Open the image
    img_in = cv2.imread("4mtshuffle.png", 1)

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
def Encrypt():
    filename = CONFIG.SOURCE
    img = cv2.imread(filename, 1)
    dim = img.shape

    timer = np.zeros(5)
    overall_time = time.time()

    # Check image dimensions
    timer[0] = overall_time
    if dim[0]!=dim[1]:
        N = min(dim[0], dim[1])
        img = cv2.resize(img,(N,N))
        dim = img.shape

    # Perform histogram equalization
    imgEQ = HistEQ(img)
    timer[0] = time.time() - timer[0]
    cv2.imwrite("2histeq.png", imgEQ)
    imgAr = imgEQ

    timer[1] = time.time()
    # Compute hash of imgEQ and write to text file
    imghash = sha2Hash("2histeq.png")
    timer[1] = time.time() - timer[1]
    f = open("hash.txt","w+")
    f.write(str(imghash))
    f.close()

    #Clear catmap debug files
    if CONFIG.DEBUG_CATMAP:
        files = os.listdir("catmap")
        for f in files:
            os.remove(os.path.join("catmap", f))
    
    timer[2] = time.time()
    # Ar Phase: Cat-map Iterations
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    for i in range (1, random.randint(CONFIG.AR_MIN_ITER,CONFIG.AR_MAX_ITER)):
        cuda.memcpy_htod(gpuimgIn, imgAr_In)
        func = mod.get_function("ArCatMap")
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        cuda.memcpy_dtoh(imgAr_In, gpuimgOut)
        # Write intermediate files if debugging is enabled
        if CONFIG.DEBUG_CATMAP:
            imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
            cv2.imwrite("catmap\\iteration " + str(i) + ".png", imgAr)
    imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
    timer[2] = time.time() - timer[2]
    cv2.imwrite("3catmap.png", imgAr)

    # MT Phase: Intra-column pixel shuffle
    timer[3] = time.time()
    imgMT = MTShuffle(imgAr, imghash)
    timer[3] = time.time() - timer[3]
    cv2.imwrite("4mtshuffle.png", imgMT)

    timer[4] = time.time()
    # Fractal XOR Phase
    imgFr = FracXor(imghash)
    timer[4] = time.time() - timer[4]
    cv2.imwrite("5imgfractal.png", imgFr)
    overall_time = time.time() - overall_time

    if CONFIG.DEBUG_TIMER:
        print("Pre-processing completed in " + str(timer[0]) +"s")
        print("Hashing completed in " + str(timer[1]) +"s")
        print("Arnold Mapping completed in " + str(timer[2]) +"s")
        print("MT Shuffle completed in " + str(timer[3]) +"s")
        print("Fractal XOR completed in " + str(timer[4]) +"s")
        print("\nEncryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time\n")

    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()