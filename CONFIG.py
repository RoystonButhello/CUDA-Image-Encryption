'''Configure operation of the encryption algorithms via this file'''
import os

# Note:- Use absolute path when executing via VSC, use relative path when executing via IDLE
PATH = os.path.dirname(os.path.abspath( __file__ )) + "\\"

#Flags
DEBUG_HISTEQ = False
DEBUG_CATMAP = False
DEBUG_TIMER = True

#Constants
MASK_BITS = 16
AR_MIN_ITER = 100   #Min. catmap iterations
AR_MAX_ITER = 110   #Max. catmap iterations
BUFF_SIZE = 65536   #Compute hash in 64kB chunks
SOURCE = "raytracer480.png" #Input image