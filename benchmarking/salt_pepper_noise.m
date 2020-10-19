pkg load image
img_path = "baboon_1024_1024_ENC.png"
img = imread(img_path);
noisy_img = imnoise(img,'salt & pepper',0.05);
imwrite(noisy_img,"baboon_1024_1024_ENC_diffused.png");


