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
    # Inverse Ar Phase: Cat-map Iterations
    cv2.imwrite("8output.png", imgAr)
    while (cf.sha2Hash("8output.png")!=srchash):
        imgAr = cf.ArCatMap(imgAr)
        cv2.imwrite("8output.png", imgAr)
    timer[2] = time.time() - timer[2]
    overall_time = time.time() - overall_time
    
    # Print timing statistics
    if CONFIG.DEBUG_TIMER:
        print("Fractal XOR Inversion completed in " + str(timer[0]) +"s")
        print("MT Unshuffle completed in " + str(timer[1]) +"s")
        print("Arnold Mapping completed in " + str(timer[2]) +"s")
        print("Decryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()