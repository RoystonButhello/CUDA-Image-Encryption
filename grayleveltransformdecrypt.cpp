#include <iostream> /*For IO*/
#include <cstdio>   /*For printf()*/
#include <cstdint>  /*For standard variable support*/
#include <cstdlib>  /*For malloc()*/
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "include/randomfunctions.hpp"
#include "include/selfxorfunctions.hpp"
//#include "include/kernel.hpp"

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
  
  std::clock_t img_load_start = std::clock();
  image=imread("airplane_encrypted.png",IMREAD_COLOR);
  std::clock_t img_load_end = std::clock();  
  time_array[0] = (1000.0 * (img_load_end - img_load_start)) / CLOCKS_PER_SEC;

  if(!image.data)
  {
    cout<<"Image not found \nExiting...";
    exit(0);  
  } 

  if(RESIZE_TO_DEBUG==1)
  {
    cv::resize(image,image,cv::Size(1024,1024),CV_INTER_LANCZOS4);
  }
  
  m=(uint32_t)image.rows;
  n=(uint32_t)image.cols;
  
  total=m*n;
  cout<<"\nRows = "<<m;
  cout<<"\nColumns = "<<n;
  cout<<"\nTotal = "<<total;  
  
  /*CPU Declarations*/
  uint16_t *random_array = (uint16_t*)malloc(sizeof(uint16_t) * total * 3); 
  uint8_t  *img_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);   
  double *x = (double*)malloc(sizeof(double) * total * 3);
  double *y = (double*)malloc(sizeof(double) * total * 3);
  
  /*GPU Declarations*/
  uint16_t *gpuRandomArray;
  uint8_t  *gpuimgVec;
   

  x[0] = 0.1;
  y[0] = 0.1;
  myu  = 0.9;  

  std::clock_t c_start = std::clock();
  twodLogisticAdjustedSineMap(x,y,random_array,myu,total);
  std::clock_t c_end = std::clock();
  time_array[1] = (1000.0 * (c_end-c_start)) / CLOCKS_PER_SEC;
  
  
  
  /*if(DEBUG_VECTORS==1)
  { 
    cout<<"\nrandom_array = ";
    for(int i = 0; i < number; ++i)
    {
      printf(" %d",random_array[i]);
    }
  }*/
  
  //Flatten image
  std::clock_t flatten_image_start = std::clock();
  flattenImage(image,img_vec);
  std::clock_t flatten_image_end = std::clock();
  time_array[2] = (1000.0 * (flatten_image_end-flatten_image_start)) / CLOCKS_PER_SEC;
 
  /*if(DEBUG_VECTORS==1)
  {
    cout<<"\noriginal img_vec= ";
    for(int i=0;i < total*3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
  }*/
  
  /*float time;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);*/
 
  /*cudaMalloc((void**)&gpuRandomArray, total * 3 * sizeof(uint16_t));
  cudaMalloc((void**)&gpuimgVec, total * 3 * sizeof(uint8_t));  
  
  cudaMemcpy(gpuRandomArray, random_array, total * 3 * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuimgVec, img_vec, total * 3 * sizeof(uint16_t), cudaMemcpyHostToDevice);

  dim3 gray_level_transform_grid(m*n,1,1);
  dim3 gray_level_transform_block(3,1,1);*/  
  
  
  
  std::clock_t gray_level_enc_start = std::clock();
  //run_grayLevelTransform(gpuimgVec,gpuRandomArray,gray_level_transform_grid,gray_level_transform_block);
  grayLevelTransform(img_vec,random_array,total);
  std::clock_t gray_level_enc_end = std::clock();
  time_array[3] = (1000.0 * (gray_level_enc_end-gray_level_enc_start)) / CLOCKS_PER_SEC;
  
  //cudaMemcpy(img_vec,gpuimgVec,total * 3 * sizeof(uint8_t),cudaMemcpyDeviceToHost);
  /*cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);*/
  
  
  if(DEBUG_IMAGES==1)
  {
    std::clock_t img_reshape_start = std::clock();
    cv::Mat img_reshape(m,n,CV_8UC3,img_vec);
    std::clock_t img_reshape_end = std::clock();
    time_array[4] = (1000.0 * (img_reshape_end - img_reshape_start)) / CLOCKS_PER_SEC;
    
    std::clock_t img_write_start = std::clock();
    cv::imwrite("airplane_decrypted.png",img_reshape);
    std::clock_t img_write_end = std::clock();
    time_array[5] = (1000.0 * (img_write_end - img_write_start)) / CLOCKS_PER_SEC;
  }  

  
  
  /*if(DEBUG_VECTORS==1)
  {
    cout<<"\nimg_vec after decryption= ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
  }*/
  
  for(int i = 0; i < 6; ++i)
  {
    total_time=total_time + time_array[i];
  }  

  printf("\n Load image = %Lf ms", time_array[0]);
  printf("\n Generate random array = %Lf ms",time_array[1]);
  printf("\n Flatten image = %Lf ms",time_array[2]);
  printf("\n Gray Level Transform decrypt call = %Lf ms",time_array[3]);
  //printf("\n Gray Level Transform Kernel = %f ms",time);
  printf("\n Reshape image = %Lf ms",time_array[4]);
  printf("\n Write Image = %Lf ms",time_array[5]);
  printf("\n Total time = %Lf ms",total_time);
  
  return 0;
}

