import os

# Note:- Use absolute path when executing via VSC, use relative path when executing via IDLE
PATH = os.path.dirname(os.path.abspath( __file__ )) + "\\"

#Flags
DEBUG_HISTEQ = False
DEBUG_CATMAP = True
DEBUG_TIMER = True

#Constants
MASK_BITS = 8 #no. of bits + 1
AR_MIN_ITER = 90
AR_MAX_ITER = 100