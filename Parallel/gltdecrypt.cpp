#include <iostream> /*For IO*/
#include <cstdio>   /*For printf()*/
#include <cstdint>  /*For standard variable support*/
#include <cstdlib>  /*For malloc()*/
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "include/randomfunctions.hpp"
#include "include/selfxorfunctions.hpp"
#include "include/kernel.hpp"



using namespace std;
using namespace cv;

int main()
{
  //int lower_limit=1,upper_limit=8000;
  uint32_t m=0,n=0,total=0;
  double myu=0.0;
  long double total_time=0.00;
  long double time_array[6]; 
  
  cv::Mat image;
  image = cv::imread("airplane_encrypted.png",IMREAD_COLOR);   

  if(!image.data)
  {
    cout<<"\nCould not find image \nExiting...";
    exit(0);
  }
  
  if(RESIZE_TO_DEBUG==1)
  {
    cv::resize(image,image,cv::Size(1024,1024));
  }  
  
  m = (uint32_t)image.rows;
  n = (uint32_t)image.cols;
  total = m * n;
  cout<<"\nRows = "<<m;
  cout<<"\nColumns = "<<n;
  cout<<"\nTotal = "<<total;
  
  
  
  /*CPU Vectors*/
  uint8_t *img_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);  
  uint16_t *random_array = (uint16_t*)malloc(sizeof(uint16_t) * total * 3);
  double *x = (double*)malloc(sizeof(double) * total * 3);
  double *y = (double*)malloc(sizeof(double) * total * 3);   

  /*GPU Vectors*/
  uint16_t *gpuRandomArray;
  uint8_t *gpuimgVec;
  
  
  
  /*Generate Random Vector*/
  x[0] = 0.1;
  y[0] = 0.1;
  myu  = 0.9; 
  twodLogisticAdjustedSineMap(x,y,random_array,myu,total);  
  //MTMap(random_array,total,1,255,123456789);  

  /*Flatten image*/
  flattenImage(image,img_vec);
 
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\nEncrypted image vector =";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
  }
  
  
  
  cout<<"\nBefore cudaMalloc";
  cudaMalloc((void**)&gpuRandomArray,total * 3 * sizeof(uint16_t));
  cudaMalloc((void**)&gpuimgVec, total * 3 * sizeof(uint8_t));
  cout<<"\nAfter cudaMalloc";
  
  cout<<"\nBefore cudaMemcpy";
  cudaMemcpy(gpuRandomArray,random_array,total * 3 * sizeof(uint16_t),cudaMemcpyHostToDevice);
  cudaMemcpy(gpuimgVec,img_vec,total * 3 * sizeof(uint8_t),cudaMemcpyHostToDevice);
  cout<<"\nAfter cudaMemcpy";
  
  dim3 gray_level_transform_grid(m * n,1,1);
  dim3 gray_level_transform_block(3,1,1);
  
  run_grayLevelTransform(gpuimgVec,gpuRandomArray,gray_level_transform_grid,gray_level_transform_block);
  //run_grayLevelTransform(gpuimgVec,gpuRandomArray,gray_level_transform_grid,gray_level_transform_block);
  cudaMemcpy(img_vec,gpuimgVec,total * 3 * sizeof(uint8_t),cudaMemcpyDeviceToHost);
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\nDecrypted image vector= ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
  }
  
  
  if(DEBUG_IMAGES == 1)
  {
    cv::Mat img_reshape(m,n,CV_8UC3,img_vec);
    cv::imwrite("airplane_decrypted.png",img_reshape);
  }
  
  return 0;
}

