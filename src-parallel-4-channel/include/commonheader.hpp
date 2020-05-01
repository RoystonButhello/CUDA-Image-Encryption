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
#include <cstdbool> /*For boolean variables*/
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
  static inline void flattenImage(cv::Mat image, uint8_t *&img_vec, uint32_t channels);
  static inline void printImageContents(cv::Mat image, uint32_t channels);
  static inline uint8_t checkOverflow(uint16_t  number_1,uint16_t number_2);

  static inline void show_ieee754 (double f);
  static inline void print_int_bits(int num);
  static inline uint16_t get_n_mantissa_bits_safe(double f,int number_of_bits);
  

  static inline void writeVectorToFile32(uint32_t *&vec,int length,std::string filename);
  static inline void writeVectorToFile8(uint8_t *&vec,int length,std::string filename);
  
  static inline void printArray8(uint8_t *&arr,int length);
  static inline void printArray16(uint16_t *&arr,int length);
  static inline void printArray32(uint32_t *&arr,int length);
  static inline void printArrayDouble(double *&arr,int length);
  
  //static inline uint32_t getLast8Bits(uint32_t number);
  //static inline uint64_t getManipulatedSystemTime();
  //static inline uint32_t getLargestPrimeFactor(uint8_t n); 
  //static inline void generatePRNG(uint8_t *&random_array,uint32_t alpha,uint32_t manip_sys_time);
  
  static inline uint32_t getRandomUnsignedInteger32(uint32_t lower_bound,uint32_t upper_bound);
  static inline int getRandomInteger(int lower_bound,int upper_bound);
  static inline uint8_t getRandomUnsignedInteger8(uint8_t lower_bound,uint8_t upper_bound);
  static inline double getRandomDouble(double lower_limit,double upper_limit);
  
  static inline config::ChaoticMap mapAssigner(int lower_limit, int upper_limit);
  
  static inline void rowColLUTGen(uint32_t *&rowSwapLUT,uint32_t *&rowRandVec,uint32_t *&colSwapLUT,uint32_t *&colRandVec,uint32_t m,uint32_t n);
  static inline void genLUTVec(uint32_t *&lutVec,uint32_t n);
  
  static inline void initializeMapParameters(config::lm lm_parameters[],config::lma lma_parameters[],config::slmm slmm_parameters[],config::lasm lasm_parameters[],config::lalm lalm_parameters[],config::mt mt_parameters[],int number_of_rounds);
  static inline void assignMapParameters(config::lm lm_parameters[],config::lma lma_parameters[],config::slmm slmm_parameters[],config::lasm lasm_parameters[],config::lalm lalm_parameters[],config::mt mt_parameters[],int number_of_rounds);
  static inline void displayMapParameters(config::lm lm_parameters[],config::lma lma_parameters[],config::slmm slmm_parameters[],config::lasm lasm_parameters[],config::lalm lalm_parameters[],config::mt mt_parameters[],int number_of_rounds);

   
  
  static inline void flattenImage(cv::Mat image,uint8_t *&img_vec,uint32_t channels)
  {
    cout<<"\nIn flattenImage";
    uint16_t m=0,n=0;
    uint32_t total=0;
    m=(uint16_t)image.rows;
    n=(uint16_t)image.cols;
    total=m*n;
    image=image.reshape(1,1);
    for(int i=0;i<total * channels;++i)
    {
      img_vec[i]=image.at<uint8_t>(i);
    }
  }

  static inline void printImageContents(cv::Mat image,uint32_t channels)
  {
    //cout<<"\nIn printImageContents";
    //cout<<"\nImage Matrix=";
    for(uint32_t i=0;i<image.rows;++i)
    { 
      printf("\n");
      //printf("\ni=%d\t",i);
      for(uint32_t j=0;j<image.cols;++j)
      {
         for(uint32_t k=0;k < channels;++k)
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
      printf(" %d",arr[i]);
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
  
  static inline void printArrayDouble(double *&arr,int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf(" %f",arr[i]);
    }
  }
  
  /*static inline uint32_t getLast8Bits(uint32_t number)
  {
    //cout<<"\nIn getLast8Bits";
    uint32_t  result=number & 0xFF;
    return result;

  }*/

  /*static inline uint64_t getManipulatedSystemTime()
  {
    //cout<<"\nIn getManipulatedSystemTime";
    uint64_t microseconds_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    uint64_t manip_sys_time=(microseconds_since_epoch%255);  
    //printf("\n\n\nMicroseconds since epoch=%ld\t",microseconds_since_epoch);
    return manip_sys_time;
  }*/

  /*static inline uint32_t getLargestPrimeFactor(uint32_t number)
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

  }*/

  /*static inline void generatePRNG(uint8_t *&random_array,uint32_t alpha,uint32_t manip_sys_time)
  {
    //cout<<"\nIn generatePRNG";
    uint32_t largest_prime_factor=0;

    for(uint32_t i=0;i<256;++i)
    {
    
      manip_sys_time=manip_sys_time+alpha;
      largest_prime_factor=getLargestPrimeFactor(manip_sys_time);
    
      printf("\n\nlargest_prime_factor = %d",largest_prime_factor);
      printf("\n\nmanip_sys_time = %d",manip_sys_time);
      printf("\n\nalpha = %d",alpha);
      printf("\n\nrandom_number= %d",random_number);
    
      random_array[i]=(getLast8Bits(largest_prime_factor*manip_sys_time));
      //printf("\n\nrandom_array[%d]= %d",i,random_array[i]);
    }
  }*/
  
  static inline uint8_t getRandomUnsignedInteger8(uint8_t lower_bound,uint8_t upper_bound)
  {
      std::random_device r;
      std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
      mt19937 seeder(seed);
      uniform_int_distribution<uint8_t> intGen(lower_bound, upper_bound);
      uint8_t alpha=intGen(seeder);
      return alpha;
  }

  static inline uint32_t getRandomUnsignedInteger32(uint32_t lower_bound,uint32_t upper_bound)
  {
      //cout<<"\nIn getSeed";
      std::random_device r;
      std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
      mt19937 seeder(seed);
      uniform_int_distribution<uint32_t> intGen(lower_bound, upper_bound);
      uint32_t alpha=intGen(seeder);
      return alpha;
  }
  
  static inline int getRandomInteger(int lower_bound,int upper_bound)
  {
      //cout<<"\nIn getSeed";
      std::random_device r;
      std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
      mt19937 seeder(seed);
      uniform_int_distribution<int> intGen(lower_bound, upper_bound);
      uint32_t alpha=intGen(seeder);
      return alpha;
  }  

  static inline double getRandomDouble(double lower_limit,double upper_limit)
  {
     std::random_device r;
     std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
     mt19937 seeder(seed);
     //uniform_int_distribution<int> intGen(1, 32);
     uniform_real_distribution<double> realGen(lower_limit, upper_limit);   
     auto randnum=realGen(seeder);
     return randnum;
  }
  
  static inline config::ChaoticMap mapAssigner(int lower_limit, int upper_limit)
  {
    config::ChaoticMap chaotic_map;
    chaotic_map = (config::ChaoticMap)getRandomInteger(lower_limit,upper_limit);
    return chaotic_map;
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
  
  static inline void initializeMapParameters(config::lm lm_parameters[],config::lma lma_parameters[],config::slmm slmm_parameters[],config::lasm lasm_parameters[],config::lalm lalm_parameters[],config::mt mt_parameters[],int number_of_rounds)
  {
    for(int i = 0; i < number_of_rounds; ++i)
    {
      cout<<"\nROUND "<<i;
      //Initializing all parameters to zero
      lm_parameters[i].x_init = 0.00;
      lm_parameters[i].y_init = 0.00;
      lm_parameters[i].r = 0.00;
      
      lma_parameters[i].x_init = 0.00;
      lma_parameters[i].y_init = 0.00;
      lma_parameters[i].myu1 = 0.00;
      lma_parameters[i].myu2 = 0.00;
      lma_parameters[i].lambda1 = 0.00;
      lma_parameters[i].lambda2 = 0.00;
      
      slmm_parameters[i].x_init = 0.00;
      slmm_parameters[i].y_init = 0.00;
      slmm_parameters[i].alpha = 0.00;
      slmm_parameters[i].beta = 0.00;
      
      lasm_parameters[i].x_init = 0.00;
      lasm_parameters[i].y_init = 0.00;
      lasm_parameters[i].myu = 0.00;
      
      lalm_parameters[i].x_init = 0.00;
      lalm_parameters[i].y_init = 0.00;
      lalm_parameters[i].myu = 0.00;
      
      mt_parameters[i].seed_1 = 0;
      mt_parameters[i].seed_2 = 0;
      mt_parameters[i].seed_3 = 0;
      mt_parameters[i].seed_4 = 0;
      
    }
    mt_parameters[0].seed_5 = 0;
  }
  
  static inline void assignMapParameters(config::lm lm_parameters[],config::lma lma_parameters[],config::slmm slmm_parameters[],config::lasm lasm_parameters[],config::lalm lalm_parameters[],config::mt mt_parameters[],int number_of_rounds)
  {
    //Assigning random values to parameters
    for(int i = 0; i < number_of_rounds; ++i)
    {
      cout<<"\nROUND "<<i;
      lm_parameters[i].x_init = getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
      lm_parameters[i].y_init = getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
      lm_parameters[i].r = getRandomDouble(R_LOWER_LIMIT,R_UPPER_LIMIT);
      
      lma_parameters[i].x_init = getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
      lma_parameters[i].y_init = getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
      lma_parameters[i].myu1 =  getRandomDouble(MYU1_LOWER_LIMIT,MYU1_UPPER_LIMIT);
      lma_parameters[i].myu2 = getRandomDouble(MYU2_LOWER_LIMIT,MYU2_UPPER_LIMIT);
      lma_parameters[i].lambda1 = getRandomDouble(LAMBDA1_LOWER_LIMIT,LAMBDA1_UPPER_LIMIT);
      lma_parameters[i].lambda2 = getRandomDouble(LAMBDA2_LOWER_LIMIT,LAMBDA2_UPPER_LIMIT);
      
      slmm_parameters[i].x_init = getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
      slmm_parameters[i].y_init = getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
      slmm_parameters[i].alpha = getRandomDouble(ALPHA_LOWER_LIMIT,ALPHA_UPPER_LIMIT);
      slmm_parameters[i].beta = getRandomDouble(BETA_LOWER_LIMIT,BETA_UPPER_LIMIT);

      lasm_parameters[i].x_init = getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
      lasm_parameters[i].y_init = getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
      lasm_parameters[i].myu = getRandomDouble(MYU_LOWER_LIMIT,MYU_UPPER_LIMIT);
      
      lalm_parameters[i].x_init = getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
      lalm_parameters[i].y_init = getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
      lalm_parameters[i].myu = getRandomDouble(MYU_LOWER_LIMIT,MYU_UPPER_LIMIT);
      
      mt_parameters[i].seed_1 = getRandomInteger(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
      mt_parameters[i].seed_2 = getRandomInteger(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
      mt_parameters[i].seed_3 = getRandomInteger(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
      mt_parameters[i].seed_4 = getRandomInteger(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
      
    } 
    mt_parameters[0].seed_5 = getRandomInteger(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
  }
  
  static inline void displayMapParameters(config::lm lm_parameters[],config::lma lma_parameters[],config::slmm slmm_parameters[],config::lasm lasm_parameters[],config::lalm lalm_parameters[],config::mt mt_parameters[],int number_of_rounds)
  {
    for(int i = 0; i < number_of_rounds; ++i)
    {
      cout<<"\n\nROUND "<<i;
      printf("\nlm_parameters.x_init = %f",lm_parameters[i].x_init);
      printf("\nlm_parameters.y_init = %f",lm_parameters[i].y_init);
      printf("\nlm_parameters.r = %f",lm_parameters[i].r);
      
      printf("\n\nlma_parameters.x_init = %f",lma_parameters[i].x_init);
      printf("\nlma_parameters.y_init = %f",lma_parameters[i].y_init);
      printf("\nlma_parameters.myu1 = %f",lma_parameters[i].myu1);
      printf("\nlma_parameters.myu2 = %f",lma_parameters[i].myu2);
      printf("\nlma_parameters.lambda1 = %f",lma_parameters[i].lambda1);
      printf("\nlma_parameters.lambda2 = %f",lma_parameters[i].lambda2);
      
      printf("\n\nslmm_parameters.x_init = %f",slmm_parameters[i].x_init);
      printf("\nslmm_parameters.y_init = %f",slmm_parameters[i].y_init);
      printf("\nslmm_parameters.alpha = %f",slmm_parameters[i].alpha);
      printf("\nslmm_parameters.beta = %f",slmm_parameters[i].beta);
      
      printf("\n\nlasm parameters.x_init = %f",lasm_parameters[i].x_init);
      printf("\nlasm parameters.y_init = %f",lasm_parameters[i].y_init);
      printf("\nlasm parameters.myu = %f",lasm_parameters[i].myu);
      
      printf("\n\nlalm parameters.x_init = %f",lalm_parameters[i].x_init);
      printf("\nlalm parameters.y_init = %f",lalm_parameters[i].y_init);
      printf("\nlalm parameters.myu = %f",lalm_parameters[i].myu);
      
      printf("\n\nmt_parameters.seed_1 = %d",mt_parameters[i].seed_1);
      printf("\nmt_parameters.seed_2 = %d",mt_parameters[i].seed_2);
      printf("\nmt_parameters.seed_3 = %d",mt_parameters[i].seed_3);
      printf("\nmt_parameters.seed_4 = %d",mt_parameters[i].seed_4);
      

    }
      printf("\nmt_parameters.seed_5 = %d",mt_parameters[0].seed_5);
  }
}

#endif

