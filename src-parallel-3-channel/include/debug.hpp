#ifndef DEBUG_H
#define DEBUG_H
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

/*Function Prototypes*/
void checkImageVectors(int *plain_img_vec,int *decrypted_img_vec,uint16_t total);
void checkImageMatrices(cv::Mat encrypted_image,cv::Mat decrypted_image);


void checkImageVectors(int *plain_img_vec,int *decrypted_img_vec,uint16_t total)
{
  int cnt=0,gt255=0;
  for(int i=0;i<total*3;++i)
  {
    if(decrypted_img_vec[i]-plain_img_vec[i]!=0)
    {
      ++cnt;
    }
    
    if(decrypted_img_vec[i]>255||plain_img_vec[i]>255)
    {
      ++gt255;
    }
    
  }
  printf("\nNumber of vector differences= %d",cnt);
  printf("\nNumber of gt255= %d",gt255);
}

void checkImageMatrices(cv::Mat encrypted_image,cv::Mat decrypted_image)
{
  int cnt=0;
  for(int i=0;i<encrypted_image.rows;++i)
  {
    for(int j=0;j<encrypted_image.cols;++j)
    {
      for(int k=0;k<3;++k)
      {
         if(decrypted_image.at<Vec3b>(i,j)[k]-encrypted_image.at<Vec3b>(i,j)[k]!=0)
         {
           ++cnt;
         }        
      }
    }
  } 
  printf("\nNumber of matrix differences= %d",cnt);
}

#endif

