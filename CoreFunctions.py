import cv2                  #OpenCV
import os                   #Path setting and file-retrieval
import glob                 #File counting
import random               #Obviously neccessary
import numpy as np          #See above
import CONFIG               #Module with Debug flags and other constants
import time                 #Literally just timing
import hashlib              #For SHA256

os.chdir(CONFIG.PATH)


def sha2alt(img,N):
  time_array=numpy.zeros([4])
  cv2.resize(img,(N,N))
  data = numpy.array(img_in)
  flattened = data.flatten()
  st_1=time.time()
  hash_flattened=sha256()
  hash_flattened.update(flattened)
  
  #hash_flattened.digest()
  final_hash=int(hash_flattened.hexdigest(),16)
  time_array[0]=time.time()-st1
  hash_str=str(final_hash)
  hash_file=open("hash.txt","w+")
  n=0
  n=hash_file.write(hash_str)
  hash_file.close()
  print("\n time for hashing= "+str(time_array[0]))

# Arnold's Cat Map
def ArCatMap(img_in):
    dim = img_in.shape
    N = dim[0]
    img_out = np.zeros([N, N, dim[2]])

    for x in range(N):
        for y in range(N):
            img_out[x][y] = img_in[(2*x+y)%N][(x+y)%N]

    return img_out

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
def FracXor(filename, imghash):
    #Open the image
    img_in = cv2.imread(filename, 1)

    #Select a file for use based on hash
    fileCount = len(glob.glob1("fractals","*.png"))
    fracID = (imghash % fileCount) + 1
    filename = "fractals" + CONFIG.SEPARATOR + str(fracID) + ".png"
    #Read the file, resize it, then XOR
    fractal = cv2.imread(filename, 1)
    dim = img_in.shape
    fractal = cv2.resize(fractal,(dim[0],dim[1]))
    img_out = cv2.bitwise_xor(img_in,fractal)

    return img_out

#Clear catmap debug files
def CatmapClear():
    files = os.listdir("catmap")
    for f in files:
        os.remove(os.path.join("catmap", f))