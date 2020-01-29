
import os
import platform
# Note:- Use absolute path when executing via VSC, use relative path when executing via IDLE

#Directory or file separator By default set to Windows
SEPARATOR="\\"

if os.name=='nt':
  SEPARATOR="\\"
  print("\nYou are running "+str(platform.system())+" "+str(platform.release()))
  PATH = os.path.dirname(os.path.abspath( __file__ )) + SEPARATOR
  print("\nCurrent PATH= "+PATH)

elif os.name=='posix':
  SEPARATOR="/"
  PATH = os.path.dirname(os.path.abspath( __file__ )) + SEPARATOR
  print("\nYou are running "+str(platform.system())+" "+str(platform.release()))
  print("\nCurrent PATH= "+PATH)

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


'''Configure operation of the encryption algorithms via this file'''


# Note:- Use absolute path when executing via VSC, use relative path when executing via IDLE
PATH = os.path.dirname(os.path.abspath( __file__ )) + SEPARATOR
TEMP = "temp"+SEPARATOR
SRC  = "input"+SEPARATOR

IN      =  SRC + "raytracer480.png"  # Input image
HASH    = TEMP + "hash.txt"          # Image Hash  
HISTEQ  = TEMP + "2histeq.png"       # Histogram-equalized square Image
ARMAP   = TEMP + "3armap.png"        # Arnold-mapped Image
MT      = TEMP + "4mtshuffle.png"    # MT-Shuffled Image
XOR     = TEMP + "5xorfractal.png"   # Fractal-XOR'd Image
UnXOR   = TEMP + "6xorunfractal.png" # Fractal-UnXOR'd Image
UnMT    = TEMP + "7mtunshuffle.png"  # MT-UnShuffled Image
ARTEMP  = TEMP + "catmap"+ SEPARATOR +"iteration" # Debug files for ArMap intermediate results
OUT     = "out.png"                  # Output Image (and ArUnMap intermediate)


