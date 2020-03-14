#include <iostream> /*For IO*/
#include <cstdio>   /*For printf()*/
#include <cstdint>  /*For standard variable support*/
#include <cstdlib>  /*For malloc()*/
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "include/randomfunctions.hpp"
#include "include/selfxorfunctions.hpp"



using namespace std;
using namespace cv;

int main()
{
  //int lower_limit=1,upper_limit=8000;
  uint32_t m=0,n=0,total=0;
  double myu=0.0;
  
  cv::Mat image;
  
  image=imread("images/airplane.png",IMREAD_COLOR);
  if(!image.data)
  {
    cout<<"Image not found \nExiting...";
    exit(0);  
  } 

  if(RESIZE_TO_DEBUG==1)
  {
    cv::resize(image,image,cv::Size(50,50),CV_INTER_LANCZOS4);
  }
  
  m=(uint32_t)image.rows;
  n=(uint32_t)image.cols;
  
  total=m*n;
  cout<<"\nRows = "<<m;
  cout<<"\nColumns = "<<n;
  cout<<"\nTotal = "<<total;  
  
  /*Declarations*/
  uint16_t *random_array = (uint16_t*)malloc(sizeof(uint16_t) * total * 3); 
  uint8_t  *img_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);   
  double *x = (double*)malloc(sizeof(double) * total * 3);
  double *y = (double*)malloc(sizeof(double) * total * 3);
  
  x[0] = 0.1;
  y[0] = 0.1;
  myu  = 0.9;  

  std::clock_t c_start = std::clock();
  twodLogisticAdjustedSineMap(x,y,random_array,myu,total);
  std::clock_t c_end = std::clock();
  long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  
  printf("\ntime elapsed= %Lf ms",time_elapsed_ms);
  
  
  if(DEBUG_VECTORS==1)
  { 
    /*cout<<"\nrandom_array = ";
    for(int i = 0; i < number; ++i)
    {
      printf(" %d",random_array[i]);
    }*/
  }
  
  //Flatten image
  flattenImage(image,img_vec);
  
  if(DEBUG_VECTORS==1)
  {
    cout<<"\noriginal img_vec= ";
    for(int i=0;i < total*3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
  }
  
  grayLevelTransform(img_vec,random_array,total);
  
  if(DEBUG_IMAGES==1)
  {
    cv::Mat img_reshape(m,n,CV_8UC3,img_vec);
    cv::imwrite("images/airplane_gray_level_transform_encrypted.png",img_reshape);
  }  

  if(DEBUG_VECTORS==1)
  {
    cout<<"\nimg_vec after encryption= ";
    for(int i=0; i < total * 3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
  }
  
  grayLevelTransform(img_vec,random_array,total);
  
  if(DEBUG_IMAGES==1)
  {
    cv::Mat img_reshape(m,n,CV_8UC3,img_vec);
    cv::imwrite("images/airplane_gray_level_transform_decrypted.png",img_reshape);
  }
  
  if(DEBUG_VECTORS==1)
  {
    cout<<"\nimg_vec after decryption= ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
  }
  
  return 0;
}

