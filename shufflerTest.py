import os, random, cv2, CONFIG, numpy as np   

PATH = os.path.dirname(os.path.abspath( __file__ )) + "\\"
os.chdir(PATH)

#Generate 64-bit Integer hash based on rough horizontal gradient
def gradientHash(img, hashSize=8):
    # Resize image; add extra column to compute the horizontal gradient
    imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgScaled = cv2.resize(imgBW,(hashSize+1,hashSize))

    # Compute horizontal gradient b/w adjacent pixels
    diffMat = imgScaled[:, 1:] > imgScaled[:, :-1]

    # Convert the gradient image to a hash
    return sum([2**i for (i, v) in enumerate(diffMat.flatten()) if v])

def MTShuffle(img_in, imghash):
    mask = 2**CONFIG.MASK_BITS - 1   # Default: 8 bits
    temphash = imghash
    dim = img_in.shape
    N = dim[0]
    for j in range(N):
        random.seed(temphash & mask)
        random.shuffle(img_in[:, j])
        temphash = temphash>>CONFIG.MASK_BITS
        if temphash==0:
            temphash = imghash
    return img_in

def MTUnShuffle(img_in, imghash):
    mask = 2**CONFIG.MASK_BITS - 1   # Default: 8 bits
    temphash = imghash
    dim = img_in.shape
    N = dim[0]
    img_out = img_in.copy()

    MTmap = np.zeros(N,dtype=int)
    for i in range(1,N):
        MTmap = np.vstack((MTmap, np.full((N), i, dtype=float)))

    for j in range(N):
        random.seed(temphash & mask)
        random.shuffle(MTmap[:, j])
        temphash = temphash>>CONFIG.MASK_BITS
        if temphash==0:
            temphash = imghash
        for i in range(N):
            index = int(MTmap[i][j])
            img_out[index][j] = img_in[i][j]
    return img_out

img = cv2.imread("minray.png", 1)
dim = img.shape
if dim[0]!=dim[1]:
        N = min(dim[0], dim[1])
        img = cv2.resize(img,(N,N))
hash = gradientHash(img)
cv2.imshow('Input image', img)
img = MTShuffle(img,hash)
cv2.imshow('Shuffled image', img)
img = MTUnShuffle(img,hash)
cv2.imshow('Unshuffled image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()