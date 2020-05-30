from skimage import io
import skimage.measure
import sys

img = io.imread(sys.argv[1])
entropy = skimage.measure.shannon_entropy(img)
print("\nentropy = " + str(entropy))
