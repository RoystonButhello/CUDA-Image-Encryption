import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
from shutil import rmtree   # Directory removal
import secrets              # CSPRNG
import warnings             # Ignore integer overflow during diffusion

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

warnings.filterwarnings("ignore", category=RuntimeWarning)

#os.chdir(cfg.PATH)

# Path-check and image reading
def Init():
    # Open Image
    img = cv2.imread(cfg.ENC_IN,-1)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    return img, img.shape[0], img.shape[1]

# Generate and return rotation vector of length n containing values < m
def genRelocVec(m, n, logfile):
    # Initialize constants
    secGen = secrets.SystemRandom()
    a = secGen.randint(2,cfg.PERMINTLIM)
    b = secGen.randint(2,cfg.PERMINTLIM)
    c = 1 + a*b
    x = secGen.uniform(0.0001,1.0)
    y = secGen.uniform(0.0001,1.0)
    offset = secGen.randint(1,cfg.PERMINTLIM)
    unzero = 0.0000001

    # Log parameters for decryption
    with open(logfile, 'w+') as f:
        f.write(str(a) +"\n")
        f.write(str(b) +"\n")
        f.write(str(x) +"\n")
        f.write(str(y) +"\n")
        f.write(str(offset) + "\n")

    # Skip first <offset> values
    for i in range(offset):
        x = (x + a*y)%1 + unzero
        y = (b*x + c*y)%1 + unzero
    
    # Start writing intermediate values
    ranF = np.zeros((m*n),dtype=np.float)
    for i in range(m*n//2):
        x = (x + a*y)%1 + unzero
        y = (b*x + c*y)%1 + unzero
        ranF[2*i] = x
        ranF[2*i+1] = y
    
    # Generate column-relocation vector
    r = secGen.randint(1,m*n-n)
    exp = 10**14
    vec = np.zeros((n),dtype=np.uint16)
    for i in range(n):
        vec[i] = int((ranF[r+i]*exp)%m)

    with open(logfile, 'a+') as f:
        f.write(str(r))

    return ranF, vec

mod = SourceModule("""
    #include <stdint.h>
    __global__ void GenCatMap(uint8_t *in, uint8_t *out, uint16_t *colRotate, uint16_t *rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int InDex    = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int OutDex   = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }
  """)

def Encrypt():
    # Initiliaze timer
    timer = np.zeros(3)
    overalltime = time.perf_counter()

    # Read image
    img, m, n = Init()

    timer[0] = time.perf_counter()
    # Col-rotation | len(U)=n, values from 0->m
    P1, U = genRelocVec(m,n,cfg.P1LOG)
    timer[0] = time.perf_counter() - timer[0]

    timer[1] = time.perf_counter()
    # Row-rotation | len(V)=m, values from 0->n
    P2, V = genRelocVec(n,m,cfg.P2LOG)
    timer[1] = time.perf_counter() - timer[1]

    imgArray  = np.asarray(img).reshape(-1)
    gpuimgIn  = cuda.mem_alloc(imgArray.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArray.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArray)
    gpuU = cuda.mem_alloc(U.nbytes)
    gpuV = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(gpuU, U)
    cuda.memcpy_htod(gpuV, V)
    func = mod.get_function("GenCatMap")

    func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(m,n,1), block=(3,1,1))
    temp = gpuimgIn
    gpuimgIn = gpuimgOut
    gpuimgOut = temp
    timer[2] = time.perf_counter() - timer[2]
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(m,n,1), block=(3,1,1))
        temp = gpuimgIn
        gpuimgIn = gpuimgOut
        gpuimgOut = temp
    timer[2] = time.perf_counter() - timer[2]

    cuda.memcpy_dtoh(imgArray, gpuimgIn)
    img = (np.reshape(imgArray,img.shape)).astype(np.uint8)

    '''PERMUTATION PHASE COMPLETE'''

    cv2.imwrite(cfg.ENC_OUT, img)
    overalltime = time.perf_counter() - overalltime
    misc = overalltime - np.sum(timer)

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Target: {} ({}x{})".format(cfg.ENC_IN, n, m))
        print("U Generation:\t{0:9.7f}s ({1:5.2f}%)".format(timer[0], timer[0]/overalltime*100))
        print("V Generation:\t{0:9.7f}s ({1:5.2f}%)".format(timer[1], timer[1]/overalltime*100))
        print("Permutation:\t{0:9.7f}s ({1:7.5f}%)".format(timer[2], timer[2]/overalltime*100))
        print("Misc. ops: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overalltime*100))
        print("Net Time: \t{0:7.5f}s\n".format(overalltime))

Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
