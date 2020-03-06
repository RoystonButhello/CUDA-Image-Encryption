/*These two lines prevent the compiler from reading the same header file twice*/
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream> /*For IO*/
#include <cstdio>   /*For printf*/
#include <string>   /*For std::string()*/
#include <random>   /*For random number generation*/
#include <chrono>   /*For time*/
#include <fstream>  /*For writing to file*/ 
#include <cstdint>  /*For standard variable types*/
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <cmath>    /*For sqrt()*/ 
#include <cstdlib>  /*For exit()*/
#include <ctime>    /*For clock()*/

#define RESIZE_TO_DEBUG 1
#define DEBUG_VECTORS   0
#define DEBUG_IMAGES    1
#define PRINT_IMAGES    0
#define INIT            100

using namespace cv;
using namespace std;

/*Function Prototypes*/

/*Self XOR Transform Phase*/
static inline void flattenImage(cv::Mat image,uint8_t *&img_vec);
static inline void printImageContents(cv::Mat image);
static inline void printVectorCircular(uint8_t *&img_vec,uint16_t xor_position,uint16_t total);
static inline void xorImageEnc(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n);
static inline void xorImageDec(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n);

/*Miscellaneous*/
static inline uint8_t checkOverflow(uint16_t  number_1,uint16_t number_2);

/*Self XOR Transform Phase Starts*/
static inline void flattenImage(cv::Mat image,uint8_t *&img_vec)
{
  //cout<<"\nIn flattenImage";
  uint16_t m=0,n=0;
  uint32_t total=0;
  m=(uint16_t)image.rows;
  n=(uint16_t)image.cols;
  total=m*n;
  image=image.reshape(1,1);
  for(int i=0;i<total*3;++i)
  {
    img_vec[i]=image.at<uint8_t>(i);
  }
}

static inline void printImageContents(Mat image)
{
  //cout<<"\nIn printImageContents";
  cout<<"\nImage Matrix=";
    for(uint32_t i=0;i<image.rows;++i)
    { printf("\n");
      //printf("\ni=%d\t",i);
      for(uint32_t j=0;j<image.cols;++j)
      {
         for(uint32_t k=0;k<3;++k)
         {
          //printf("\nj=%d\t",j);
          printf("%d\t",image.at<Vec3b>(i,j)[k]); 
         } 
       }
    }
}

static inline void printVectorCircular(uint8_t *&img_vec,uint16_t xor_position,uint16_t total)
{
  //cout<<"In printCircularVector";
  cout<<"\nCircular Image Vector=";
  for(int i=xor_position;i<xor_position+(total*3);++i)
  {
    printf(" %d",img_vec[i%(total*3)]);
  }
}


static inline void xorImageEnc(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n)
{ 
   //cout<<"\nIn xorImageEnc";
   uint32_t total=m*n;
   for(int i=0;i<total*3;++i)
   {
     img_xor_vec[i]=0;
   }

   img_xor_vec[0]=img_vec[0] ^ INIT;
   //printf("\n %d = %d ^ %d",img_xor_vec[0],img_vec[0],INIT);
   for(int i=1;i<total*3;++i)
   {
     img_xor_vec[i]=img_vec[i] ^ img_xor_vec[i-1];
     //printf("\n %d = %d ^ %d",img_xor_vec[i],img_vec[i],img_xor_vec[i-1]);
   } 
}

static inline void xorImageDec(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n)
{ 
   //cout<<"\nIn xorImageDec";
   uint32_t total=m*n;
   for(int i=0;i<total*3;++i)
   {
     img_xor_vec[i]=0;
   }

   img_xor_vec[0]=img_vec[0] ^ INIT;
   //printf("\n %d = %d ^ %d",img_xor_vec[0],img_vec[0],INIT);
   for(int i=1;i<total*3;++i)
   {
     img_xor_vec[i]=img_vec[i] ^ img_vec[i-1];
     //printf("\n %d = %d ^ %d",img_xor_vec[i],img_vec[i],img_vec[i-1]);
   } 
}

/*Self XOR Transform Phase Ends*/


/*Miscellaneous Phase Starts*/
static inline uint8_t checkOverflow(uint16_t  number_1,uint16_t number_2)
{
  //cout<<"\nIn checkOverflow";
  if((number_1*number_2)>=512)
  {
    printf("\n%d , %d exceeded 512",number_1,number_2);
    return 2;
  }

  if((number_1*number_2)>=256)
  {
    printf("\n%d , %d exceeded 255",number_1,number_2);
    return 1;
  }
return 0;
}
/*Miscellaneous Phase Ends*/



#endif
