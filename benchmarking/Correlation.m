pkg load image
pkg load nan
img_name = "/home/saswat/image_data/1024_1024/lena/5/lena_unrotated_ROUND_0.png"
P =im2double(imread(img_name));

x1 = double(P(:,1:end-1)); 
y1 = double(P(:,2:end)); 
randIndex1 = randperm(numel(x1)); 
randIndex1 = randIndex1(1:1000); 
x = x1(randIndex1); 
y = y1(randIndex1); 
Ex=(1/(1000))*sum(x);
Ey=(1/(1000))*sum(y);
Dx=(1/(1000))*sum((x(:)-Ex).^2);
Dy=(1/(1000))*sum((y(:)-Ey).^2);
coxy=(1/(1000))*(sum((x-Ex).*(y-Ey)));
c_hor=coxy/(sqrt(Dx*Dy));
printf("Horizontal: %f\n",c_hor);

c_vert = corrcoef(P(1:end-1, :), P(2:end, :));
printf("Vertical: %f\n",c_vert(1,2));
c_diag = corrcoef(P(1:end-1, 1:end-1), P(2:end, 2:end));
printf("Diagonal: %f\n",c_diag(1,2));
