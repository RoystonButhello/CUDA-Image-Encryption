import sys
import cv2
import numpy as np


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows = img.shape[0]
cols = img.shape[1]
channels = img.shape[2]
total = rows * cols * channels
#img = img.reshape(rows * cols * channels)

x_1 = img[ :, 0:cols - 1, 0:2]  # All rows and columns 1 through 255 from blue plane
y_1 = img[ :, 1:cols, 0:2]    # All rows and columns 2 through 256 from blue plane

#print(x.shape)
#print(y.shape)

shape_1 = x_1.shape[0] * x_1.shape[1] * x_1.shape[2]
print("\nshape_1 = " + str(shape_1))

x_1 = x_1.reshape(shape_1)
y_1 = y_1.reshape(shape_1)

r_horiz = np.corrcoef(x_1, y_1)

print("\nhorizontal correlation = ")


x_1a = img[0:rows-1, :, 0:2]  # Rows 1 through 255 and all columns from blue plane
y_1a = img[1:rows, :, 0:2]    # Rows 2 through 256 and all columns from blue plane


shape_1a = x_1a.shape[0] * x_1a.shape[1] * x_1a.shape[2]
print("\nshape_1a = " + str(shape_1a))

x_1a = x_1a.reshape(shape_1a)
y_1a = y_1a.reshape(shape_1a)

print("\nvertical correlation = ")

r_vert = np.corrcoef(x_1a, y_1a)




x_1b = img[0:rows-1, 0:cols-1, 0:2]  # All but the last row and column
y_1b = img[1:rows, 1:cols, 0:2]      # All but the first row and column


shape_1b = x_1b.shape[0] * x_1b.shape[1] * x_1b.shape[2]
print("\nshape_1b = " + str(shape_1b))

x_1b = x_1b.reshape(shape_1b)
y_1b = y_1b.reshape(shape_1b)

print("\ndiagonal correlation = ")

r_diag = np.corrcoef(x_1b, y_1b)


print("\n\nhorizontal correlation:\n")
print(r_horiz)

print("\n\nvertical correlation:\n")
print(r_vert)

print("\n\ndiagonal correlation:\n")
print(r_diag)



#print(x_1a.shape)
#print(y_1a.shape)


#print("\nhorizontal correlation in Blue channel = " + str(r_horiz_b))