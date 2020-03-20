#include <iostream> /*For IO*/
#include <cstdio>   /*For printf()*/
#include <cstdlib>  /*For malloc()*/
#include <cstdint>  /*For standard variable support*/
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include "include/randomfunctions.hpp"
#include "include/selfxorfunctions.hpp"

void rowColLUTGen(uint16_t *&colSwapLUT,uint16_t *&colRandVec,uint16_t *&rowSwapLUT,uint16_t *&rowRandVec,uint32_t n);
void genLUTVec(uint16_t *&lutVec,uint32_t n);

void genLUTVec(uint16_t *&lutVec,uint32_t n)
{
  for(int i = 0; i < n; ++i)
  {
    lutVec[i]=i;
  }
}

void rowColLUTGen(uint16_t *&colSwapLUT,uint16_t *&colRandVec,uint16_t *&rowSwapLUT,uint16_t *&rowRandVec,uint32_t n)
{
  int jCol=0,jRow=0;
  for(int i = n - 1; i > 0; i--)
  {
    jCol = colRandVec[i] % i;
    std::swap(colSwapLUT[i],colSwapLUT[jCol]);
  } 
  
  for(int i = n - 1; i > 0; i--)
  {
    jRow = rowRandVec[i] % i;
    std::swap(rowSwapLUT[i],rowSwapLUT[jRow]);
  } 
}


int main()
{
  cv::Mat image;
  uint32_t m=0,n=0,total=0;
  int lowerLimit=0,upperLimit=0;

  image = cv::imread("airplane.png",IMREAD_COLOR);
  if(!image.data)
  {
    cout<<"\nImage not found \nExiting...";
    exit(0);  
  }
  
  if(RESIZE_TO_DEBUG==1)
  {
    cv::resize(image,image,cv::Size(4,4));
  }
  
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nOriginal image = ";
    printImageContents(image);
  }
  
  m = (uint32_t)image.rows;
  n = (uint32_t)image.cols;
  total = m * n;
  
  cout<<"\nRows = "<<m;
  cout<<"\nColumns = "<<n;
  cout<<"\nChannels = "<<image.channels();
  cout<<"\nTotal = "<<total;
  
  Mat img_enc = Mat(Size(m, n), CV_8UC3, Scalar(0, 0, 0));
  Mat img_dec = Mat(Size(m, n), CV_8UC3, Scalar(0, 0, 0)); 
  
  cout<<"\nempty_image rows = "<<img_enc.rows;
  cout<<"\nempty_image columns = "<<img_dec.cols;
  
  
  uint16_t *colSwapLUT = (uint16_t*)malloc(sizeof(uint16_t) * n);
  uint16_t *rowSwapLUT = (uint16_t*)malloc(sizeof(uint16_t) * n);
  uint16_t *rowRandVec = (uint16_t*)malloc(sizeof(uint16_t) * total * 3);
  uint16_t *colRandVec = (uint16_t*)malloc(sizeof(uint16_t) * total * 3);
  
  lowerLimit = 1;
  upperLimit = (total * 3) + 0.1 * (total * 3); 

  genLUTVec(colSwapLUT,n);
  genLUTVec(rowSwapLUT,n);
  
  if(DEBUG_VECTORS==1)
  {
    cout<<"\ncolSwapLUT before swap = ";
    for(int i = 0 ;i < n; ++i)
    {
      printf(" %d",colSwapLUT[i]);
    }
    
    cout<<"\nrowSwapLUT before swap = ";
    for(int i = 0; i < n; ++i)
    {
      printf(" %d",rowSwapLUT[i]);
    }
  } 
 
  
  MTMap(rowRandVec,total,lowerLimit,upperLimit,123456789);
  MTMap(colRandVec,total,lowerLimit,upperLimit,9);
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\nrowRandVec = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",rowRandVec[i]);
    }
    
    cout<<"\ncolRandVec = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",colRandVec[i]);
    }
  }
  
  rowColLUTGen(colSwapLUT,colRandVec,rowSwapLUT,rowRandVec,n);
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\ncolSwapLUT after swap = ";
    for(int i = 0 ;i < n; ++i)
    {
      printf(" %d",colSwapLUT[i]);
    }
    
    cout<<"\nrowSwapLUT after swap = ";
    for(int i = 0; i < n; ++i)
    {
      printf(" %d",rowSwapLUT[i]);
    }
  }  
  
  
  rowColSwapEnc(image,img_enc,rowSwapLUT,colSwapLUT);
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nempty_image after encryption = ";
    printImageContents(img_enc);
  }  
  
  
  rowColSwapDec(img_enc,img_dec,rowSwapLUT,colSwapLUT);
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nempty_image after decryption = ";
    printImageContents(img_dec);
  }
   
  
  return 0;
}
