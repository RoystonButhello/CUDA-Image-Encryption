import cv2                  #OpenCV
import os                   #Path setting and file-retrieval
import glob                 #File counting
import random               #Obviously neccessary
import numpy as np          #See above
import CONFIG               #Module with Debug flags and other constants
import time                 #Literally just timing

os.chdir(CONFIG.PATH)

#Generate 64-bit Integer hash based on rough horizontal gradient
def gradientHash(img, hashSize=8):
    # Resize image; add extra column to compute the horizontal gradient
    imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgScaled = cv2.resize(imgBW,(hashSize+1,hashSize))

    # Compute horizontal gradient b/w adjacent pixels
    diffMat = imgScaled[:, 1:] > imgScaled[:, :-1]

    # Convert the gradient image to a hash
    return sum([2**i for (i, v) in enumerate(diffMat.flatten()) if v])

# Arnold's Cat Map
def ArCatMap(img_in, num):
    dim = img_in.shape
    N = dim[0]
    img_out = img_in.copy()

    for x in range(N):
        for y in range(N):
            img_out[x][y] = img_in[(2*x+y)%N][(x+y)%N]

    if CONFIG.DEBUG_CATMAP:
        cv2.imwrite("catmap\\iteration " + str(num) + ".png", img_out)
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

# Driver function
def Decrypt():
    filename = "5imgfractal.png"

    timer = np.zeros(4)
    overall_time = time.time()

    # Read hash from sent file
    f = open("hash.txt", "r")
    srchash = int(f.read())
    f.close()

    #srchash = gradientHash(cv2.imread("2histeq.png", 1))

    #Clear catmap debug files
    if CONFIG.DEBUG_CATMAP:
        files = os.listdir("catmap")
        for f in files:
            os.remove(os.path.join("catmap", f))

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

    i = 1
    timer[2] = time.time()
    # Inverse Ar Phase: Cat-map Iterations
    while (gradientHash(imgAr)!=srchash) and (i<30):
        imgAr = ArCatMap(imgAr, i)
        i += 1
    timer[2] = time.time() - timer[2]
    cv2.imwrite("8output.png", imgAr)

    overall_time = time.time() - overall_time
    print("Fractal XOR Inversion completed in " + str(timer[0]) +"s")
    print("MT Unshuffle completed in " + str(timer[1]) +"s")
    print("Arnold Mapping completed in " + str(timer[2]) +"s")
    print("Decryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()