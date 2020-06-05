pkg load image
img_path = "lena_1kg.png"
img = imread(img_path);
noisy_img = imnoise(img,'salt & pepper',0.05);
imwrite(noisy_img,"lena_1kg_diffused.png");

