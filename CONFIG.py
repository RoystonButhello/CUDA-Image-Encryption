'''Configure operation of the encryption algorithms via this file'''
import os

# Note:- Use absolute path when executing via VSC, use relative path when executing via IDLE
PATH = os.path.dirname(os.path.abspath( __file__ )) + "\\"
TEMP = "temp\\"
SRC  = "input\\"

IN      =  SRC + "raytracer480.png"  # Input image
HASH    = TEMP + "hash.txt"          # Image Hash  
HISTEQ  = TEMP + "2histeq.png"       # Histogram-equalized square Image
ARMAP   = TEMP + "3armap.png"        # Arnold-mapped Image
MT      = TEMP + "4mtshuffle.png"    # MT-Shuffled Image
XOR     = TEMP + "5xorfractal.png"   # Fractal-XOR'd Image
UnXOR   = TEMP + "6xorunfractal.png" # Fractal-UnXOR'd Image
UnMT    = TEMP + "7mtunshuffle.png"  # MT-UnShuffled Image
ARTEMP  = TEMP + "catmap\\iteration" # Debug files for ArMap intermediate results
OUT     = "out.png"                  # Output Image (and ArUnMap intermediate)

#Flags
DEBUG_HISTEQ = False
DEBUG_CATMAP = False
DEBUG_TIMER  = True

#Constants
MASK_BITS = 16
AR_MIN_ITER = 100   #Min. catmap iterations
AR_MAX_ITER = 118   #Max. catmap iterations
BUFF_SIZE = 65536   #Compute hash in 64kB chunks