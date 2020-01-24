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
