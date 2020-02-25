'''Configure algorithm operation via this file'''
import os

# Path to set working directory
#PATH = os.path.dirname(os.path.abspath( __file__ )) + "\\"

# Input image name and extension
IMG = "lena"
EXT = ".png"

# Key paths
TEMP = "temp\\"             # Folder used to store intermediary results
SRC  = "images\\"           # Folder containing input and output

# Input/Output images
ENC_IN = IMG + EXT                # Input image
ENC_OUT= IMG + "_encrypted.png"   # Encrypted Image
DEC_OUT= IMG + "_decrypted.png"   # Decrypted Image

# Log files
P1LOG = "P1.txt"
P2LOG = "P2.txt"

# Flags
DEBUG_TIMER  = True  # Print timing statistics in console

# Control Parameters
PERM_ROUNDS= 3  # PERM_ROUNDS+1 rounds performed
PERMINTLIM = 32 # Limit when generating control paramteters for relocation vector generation
