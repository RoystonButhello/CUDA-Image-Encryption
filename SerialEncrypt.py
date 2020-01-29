import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import glob                 # File counting
import time                 # Timing Execution
import random               # Obviously neccessary
import numpy as np          # See above
import hashlib              # For SHA256
import CONFIG               # Debug flags and constants
import CoreFunctions as cf  # Common functions

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

def Encrypt():
    #Initialize Timer
    timer = np.zeros(5)
    overall_time = time.time()
    
    # Open Image
    filename = CONFIG.SOURCE
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
    timer[0] = time.time() - timer[0]
    cv2.imwrite("2histeq.png", imgEQ)
    imgAr = imgEQ

    timer[1] = time.time()
    # Compute hash of imgEQ and write to text file
    imghash = cf.sha2Hash("2histeq.png")
    timer[1] = time.time() - timer[1]
    f = open("hash.txt","w+")
    f.write(str(imghash))
    f.close()

    #Clear catmap debug files
    if CONFIG.DEBUG_CATMAP:
        cf.CatmapClear()
    
    timer[2] = time.time()
    # Ar Phase: Cat-map Iterations
    for i in range (1, random.randint(CONFIG.AR_MIN_ITER,CONFIG.AR_MAX_ITER)):
        imgAr = cf.ArCatMap(imgAr)
        if CONFIG.DEBUG_CATMAP:
            cv2.imwrite("catmap" + CONFIG.SEPARATOR + "iteration" + str(i) + ".png", imgAr)
    timer[2] = time.time() - timer[2]

    cv2.imwrite("3catmap.png", imgAr)

    timer[3] = time.time()
    # MT Phase: Intra-column pixel shuffle
    imgMT = cf.MTShuffle(imgAr, imghash)
    timer[3] = time.time() - timer[3]
    cv2.imwrite("4mtshuffle.png", imgMT)

    timer[4] = time.time()
    # Fractal XOR Phase
    imgFr = cf.FracXor("4mtshuffle.png", imghash)
    timer[4] = time.time() - timer[4]
    cv2.imwrite("5imgfractal.png", imgFr)
    overall_time = time.time() - overall_time

    # Print timing statistics
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