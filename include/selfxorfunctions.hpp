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

/*PRNG Generation Phase*/
static inline uint32_t getLast8Bits(uint32_t number);
static inline uint64_t getManipulatedSystemTime();
static inline uint32_t getLargestPrimeFactor(uint8_t n); 
static inline void generatePRNG(uint8_t *&random_array,uint32_t alpha,uint32_t manip_sys_time);
static inline uint32_t getSeed(uint8_t lower_bound,uint8_t upper_bound);

/*Self XOR Transform Phase*/
static inline void flattenImage(cv::Mat image,uint8_t *&img_vec);
static inline void printImageContents(cv::Mat image);
static inline void printVectorCircular(uint8_t *&img_vec,uint16_t xor_position,uint16_t total);
static inline void xorImageEnc(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position);
static inline void xorImageDec(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position);


/*Miscellaneous*/
static inline uint8_t checkOverflow(uint16_t  number_1,uint16_t number_2);

/*PRNG Generation Phase Starts*/
static inline uint32_t getLast8Bits(uint32_t number)
{
  //cout<<"\nIn getLast8Bits";
  uint32_t  result=number & 0xFF;
  return result;

}

static inline uint64_t getManipulatedSystemTime()
{
  //cout<<"\nIn getManipulatedSystemTime";
  uint64_t microseconds_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  uint64_t manip_sys_time=(microseconds_since_epoch%255);  
  //printf("\n\n\nMicroseconds since epoch=%ld\t",microseconds_since_epoch);
  return manip_sys_time;
}

static inline uint32_t getLargestPrimeFactor(uint32_t number)
{
   //cout<<"\nIn getLargestPrimeFactor";
   int i=0;
   for (i = 2; i <= number; i++) {
            if (number % i == 0) {
                number /= i;
                i--;
            }
        }

 //cout<<"\ni= "<<i;
 return i;

}

static inline void generatePRNG(uint8_t *&random_array,uint32_t alpha,uint32_t manip_sys_time)
{
  //cout<<"\nIn generatePRNG";
  uint32_t largest_prime_factor=0;

  for(uint32_t i=0;i<256;++i)
  {
    
    manip_sys_time=manip_sys_time+alpha;
    largest_prime_factor=getLargestPrimeFactor(manip_sys_time);
    
    /*printf("\n\nlargest_prime_factor = %d",largest_prime_factor);
    printf("\n\nmanip_sys_time = %d",manip_sys_time);
    printf("\n\nalpha = %d",alpha);
    printf("\n\nrandom_number= %d",random_number);*/
    
    random_array[i]=(getLast8Bits(largest_prime_factor*manip_sys_time));
    //printf("\n\nrandom_array[%d]= %d",i,random_array[i]);
  }
}

static inline uint32_t getSeed(uint8_t lower_bound,uint8_t upper_bound)
{
    //cout<<"\nIn getSeed";
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    mt19937 seeder(seed);
    uniform_int_distribution<int> intGen(lower_bound, upper_bound);
    uint32_t alpha=intGen(seeder);
    return alpha;
}
/*PRNG Generation Phase Ends*/



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


static inline void xorImageEnc(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position)
{ 
   //cout<<"\nIn xorImageEnc";
   uint32_t total=m*n;
   /*for(int i=0;i<total*3;++i)
   {
     img_xor_vec[i]=0;
   }*/

   int cnt=0;
   //printf("\n %d = %d ^ %d",img_xor_vec[0],img_vec[0],INIT);
   for(int i=xor_position;i>0;--i)
   {
     ++cnt;
     img_xor_vec[i]=img_vec[i] ^ img_xor_vec[i-1];
     //printf("\n %d = %d ^ %d",img_xor_vec[i],img_vec[i],img_xor_vec[i-1]);
   } 
   img_xor_vec[0]=img_xor_vec[0] ^ img_xor_vec[(total*3)-1];
   
   for(int i=1;i<(total*3)-cnt;++i)
   {
     img_xor_vec[i]=img_vec[i] ^ img_xor_vec[i-1];
   }

}

static inline void xorImageDec(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position)
{ 
   //cout<<"\nIn xorImageDec";
   uint32_t total=m*n;
   /*for(int i=0;i<total*3;++i)
   {
     img_xor_vec[i]=0;
   }*/

   int cnt=0;
   
  
   //printf("\n %d = %d ^ %d",img_xor_vec[0],img_vec[0],INIT);
   for(int i=xor_position;i>0;--i)
   {
     ++cnt;
     img_xor_vec[i]=img_vec[i] ^ img_vec[i-1];
     //printf("\n %d = %d ^ %d",img_xor_vec[i],img_vec[i],img_vec[i-1]);
   } 

   img_xor_vec[0]=img_vec[0] ^ img_xor_vec[(total*3)-1];
   
   for(int i=1;i<=(total*3)-cnt;++i)
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

