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
#include "config.hpp"

using namespace cv;
using namespace std;

namespace common
{
  static inline void flattenImage(cv::Mat3b image,uint8_t *&img_vec);
  static inline void printImageContents(cv::Mat3b image);
  static inline uint8_t checkOverflow(uint16_t  number_1,uint16_t number_2);

  static inline void show_ieee754 (double f);
  static inline void print_int_bits(int num);
  static inline uint16_t get_n_mantissa_bits_safe(double f,int number_of_bits);

  static inline void writeVectorToFile32(uint32_t *&vec,int length,std::string filename);
  static inline void writeVectorToFile8(uint8_t *&vec,int length,std::string filename);

  static inline void printArray8(uint8_t *&arr,int length);
  static inline void printArray16(uint16_t *&arr,int length);
  static inline void printArray32(uint32_t *&arr,int length);
  
  static inline uint32_t getLast8Bits(uint32_t number);
  static inline uint64_t getManipulatedSystemTime();
  static inline uint32_t getLargestPrimeFactor(uint8_t n); 
  static inline void generatePRNG(uint8_t *&random_array,uint32_t alpha,uint32_t manip_sys_time);
  static inline uint32_t getSeed(uint8_t lower_bound,uint8_t upper_bound);
  
  static inline double getRandomNumber(double lower_limit,double upper_limit);
 static inline void rowColLUTGen(uint32_t *&rowSwapLUT,uint32_t *&rowRandVec,uint32_t *&colSwapLUT,uint32_t *&colRandVec,uint32_t m,uint32_t n);
  static inline void genLUTVec(uint32_t *&lutVec,uint32_t n);
  
  

  static inline void flattenImage(cv::Mat3b image,uint8_t *&img_vec)
  {
    cout<<"\nIn flattenImage";
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

  static inline void printImageContents(cv::Mat3b image)
  {
    //cout<<"\nIn printImageContents";
    cout<<"\nImage Matrix=";
    for(uint32_t i=0;i<image.rows;++i)
    { 
      printf("\n");
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

  /* formatted output of ieee-754 representation of float */
  static inline void show_ieee754 (double f)
  {
    union {
        double f;
        uint32_t u;
    } fu = { .f = f };
    int i = sizeof f * CHAR_BIT;

    printf ("  ");
    while (i--)
        printf ("%d ", BIT_RETURN(fu.u,i));

    putchar ('\n');
    printf (" |- - - - - - - - - - - - - - - - - - - - - - "
            "- - - - - - - - - -|\n");
    printf (" |s|      exp      |                  mantissa"
            "                   |\n\n");
  }

  //Print bits of an integer
  static inline void print_int_bits(int num)
  {   
    int x=1;
    for(int bit=(sizeof(int)*8)-1; bit>=0;bit--)
    {
      /*printf("%i ", num & 0x01);
      num = num >> 1;*/
      printf("%c",(num & (x << bit)) ? '1' : '0');
    }
  }

  static inline uint16_t get_n_mantissa_bits_safe(double f,int number_of_bits)
  {
    union {
        double f;
        uint32_t u;
    } fu = { .f = f };
    //int i = sizeof f * CHAR_BIT;
    int i=number_of_bits;
    int bit_store_32=0;
    uint8_t bit_store_8=0;
    uint16_t bit_store_16=0;
    
    //printf("\nBefore assigining any bits to bit_store_32=\n");
    //print_int_bits(bit_store_16);

    while (i--)
    {
        
        if(BIT_RETURN(fu.u,i)==1)
        {
            bit_store_16 |= 1 << i;
        }
        
    }
    
    
    //printf("\nAfter assigining bits to bit_store_32=\n");
    //print_int_bits(bit_store_16);
    

    //bit_store_8=(uint8_t)bit_store_32;
    return bit_store_16;
  }


  static inline void writeVectorToFile32(uint32_t *&vec,int length,std::string filename)
  {
    std::ofstream file(filename);
    if(!file)
    {
      cout<<"\nCould not create "<<filename<<"\nExiting...";
      exit(0);
    }

    std::string elements = std::string("");  

    for(int i = 0; i < length; ++i)
    {
      elements.append(std::to_string(vec[i]));
      elements.append("\n");
    }
    file<<elements;
    file.close();
  }

  static inline void writeVectorToFile8(uint8_t *&vec,int length,std::string filename)
  {
    std::ofstream file(filename);
    if(!file)
    {
      cout<<"\nCould not create "<<filename<<"\nExiting...";
      exit(0);
    }
  
    std::string elements = std::string("");
    for(int i = 0; i < length; ++i)
    {
      elements.append(std::to_string(vec[i]));
      elements.append("\n");
    }
  
    file<<elements;
    file.close();
  }

  static inline void printArray8(uint8_t *&arr,int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf("%d",arr[i]);
    }
  }

  static inline void printArray16(uint16_t *&arr, int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf(" %d",arr[i]);
    }
  }

  static inline void printArray32(uint32_t *&arr, int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf(" %d",arr[i]);
    }
  }

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
  
  static inline double getRandomNumber(double lower_limit,double upper_limit)
  {
     std::random_device r;
     std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
     mt19937 seeder(seed);
     //uniform_int_distribution<int> intGen(1, 32);
     uniform_real_distribution<double> realGen(lower_limit, upper_limit);   
     auto randnum=realGen(seeder);
     return (double)randnum;
  }

  static inline void rowColLUTGen(uint32_t *&rowSwapLUT,uint32_t *&rowRandVec,uint32_t *&colSwapLUT,uint32_t *&colRandVec,uint32_t m,uint32_t n)
  {
    int jCol=0,jRow=0;
    for(int i = m - 1; i > 0; i--)
    {
      jRow = rowRandVec[i] % i;
      std::swap(rowSwapLUT[i],rowSwapLUT[jRow]);
    }
  
    for(int i = n - 1; i > 0; i--)
    {
      jCol = colRandVec[i] % i;
      std::swap(colSwapLUT[i],colSwapLUT[jCol]);
    } 
  }  

  static inline void genLUTVec(uint32_t *&lut_vec,uint32_t n)
  {
    for(int i = 0; i < n; ++i)
    {
      lut_vec[i] = i;
    }
  }

}


#endif
