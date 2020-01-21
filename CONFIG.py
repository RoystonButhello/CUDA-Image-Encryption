import os

# Note:- Use absolute path when executing via VSC, use relative path when executing via IDLE
PATH = os.path.dirname(os.path.abspath( __file__ )) + "\\"

#Flags
DEBUG_HISTEQ = False
DEBUG_CATMAP = False
DEBUG_TIMER = True

#Constants
MASK_BITS = 16
AR_MIN_ITER = 50    #Min. catmap iterations
AR_MAX_ITER = 150   #Max/ catmap iterations
BUFF_SIZE = 65536   #Compute hash in 64kB chunks
SOURCE = "raytracer480.png" #Input image