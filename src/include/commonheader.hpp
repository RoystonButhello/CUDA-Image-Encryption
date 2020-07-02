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
  
  
  static inline double getRandomDouble(double lower_limit,double upper_limit);
  static inline int getRandomInteger(int lower_limit,int upper_limit);
  static inline uint32_t getRandomUnsignedInteger32(int lower_limit,int upper_limit);
  static inline uint8_t getRandomUnsignedInteger8(int lower_limit,int upper_limit);
  
 static inline void rowColLUTGen(uint16_t *&rowSwapLUT,uint16_t *&rowRandVec,uint16_t *&colSwapLUT,uint16_t *&colRandVec,uint32_t m,uint32_t n);
  static inline void genLUTVec(uint16_t *&lutVec,uint32_t n);
  static inline void initializeImageToZero(cv::Mat3b &image);
  static inline void writeParameterRecords(config::algorithm *parameter_records,char *filename,size_t number_of_records);
  static inline void readParameterRecords(config::algorithm *&parameter_records, char *filename, size_t number_of_records);
  
  

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
    //cout<<"\nImage Matrix=";
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

  static inline double getRandomDouble(double lower_limit,double upper_limit)
  {
     std::random_device r;
     std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
     mt19937 seeder(seed);
     //uniform_int_distribution<int> intGen(1, 32);
     uniform_real_distribution<double> realGen(lower_limit, upper_limit);   
     auto randnum=realGen(seeder);
     return (double)randnum;
  }
  
  static inline int getRandomInteger(int lower_limit,int upper_limit)
  {
     std::random_device r;
     std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
     mt19937 seeder(seed);
     uniform_int_distribution<int> intGen(lower_limit, upper_limit);
     auto randnum=intGen(seeder);
     return (int)randnum;
  }
  
  static inline uint32_t getRandomUnsignedInteger32(int lower_limit,int upper_limit)
  {
     std::random_device r;
     std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
     mt19937 seeder(seed);
     uniform_int_distribution<uint32_t> intGen(lower_limit, upper_limit);
     auto randnum=intGen(seeder);
     return (uint32_t)randnum;
  }
  
  static inline uint8_t getRandomUnsignedInteger8(int lower_limit,int upper_limit)
  {
     std::random_device r;
     std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
     mt19937 seeder(seed);
     uniform_int_distribution<uint8_t> intGen(lower_limit, upper_limit);
     auto randnum=intGen(seeder);
     return (uint8_t)randnum;
  }
  
  static inline void rowColLUTGen(uint16_t *&rowSwapLUT,uint16_t *&rowRandVec,uint16_t *&colSwapLUT,uint16_t *&colRandVec,uint32_t m,uint32_t n)
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

  static inline void genLUTVec(uint16_t *&lut_vec,uint32_t n)
  {
    for(int i = 0; i < n; ++i)
    {
      lut_vec[i] = i;
    }
  }
  
  static inline void initializeImageToZero(cv::Mat3b &image)
  {
    cout<<"\nIn initializeImageToZeros ";
    for(int i = 0; i < image.rows;++i)
    {
      for(int j = 0; j < image.cols; ++i)
      {
        
        image.at<Vec3b>(i,j) = 0;
        
      }
    }
  }
  
  /*static inline void writeParameterRecords(config::algorithm *parameter_records,char *filename,size_t number_of_records)
   { 
     cout<<"\nIn writeParameterRecords";
     FILE *outfile;
     outfile = fopen(filename,"w");
     if(outfile == NULL)
     {
       cout<<"\nCould not open "<<filename<<" for writing \nExiting...";
       exit(0);
     }
     
     for(int i = 0; i < number_of_records; ++i)
     { 
        printf("\n\nRECORD %d",i);
        printf("\nx_init = %f",parameter_records[i].slmm_parameters.x_init);
        printf("\ny_init = %f",parameter_records[i].slmm_parameters.y_init);
        printf("\nalpha = %f",parameter_records[i].slmm_parameters.alpha);
        printf("\nbeta = %f",parameter_records[i].slmm_parameters.beta);
        printf("\nrotation_rounds = %d",parameter_records[i].rotation_rounds);
        printf("\nswap_rounds = %d",parameter_records[i].swap_rounds);
        printf("\nseed_lut_gen_1 = %d",parameter_records[i].seed_lut_gen_1);
        printf("\nseed_lut_gen_2 = %d",parameter_records[i].seed_lut_gen_2);
        printf("\nseed_row_rotate = %d",parameter_records[i].seed_row_rotate);
        printf("\nseed_col_rotate = %d",parameter_records[i].seed_col_rotate);
        //printf("\ninteger_lower_limit = %d",parameter_records[i].integer_lower_limit);
        //printf("\ninteger_upper_limit = %d",parameter_records[i].integer_upper_limit);
        //printf("\ndouble_lower_limit = %f",parameter_records[i].double_lower_limit);
        //printf("\ndouble_upper_limit = %f",parameter_records[i].double_upper_limit);
            
     }
     
     size_t write_status = fwrite(&parameter_records, sizeof(config::algorithm), number_of_records, outfile);
     fclose(outfile);
     
     if(write_status != number_of_records)
     {
       cout<<"\nError in writing to "<<filename;
       printf("\nCould write only %ld of %ld records \nExiting...",write_status,number_of_records);
       exit(0);
     }
     
   }
   
   static inline void readParameterRecords(config::algorithm *&parameter_records, char *filename, size_t number_of_records)
   {
     cout<<"\nIn readParameterRecords";
     FILE *infile;
     infile = fopen(filename,"r");
     if(infile == NULL)
     {
       cout<<"\nCould not open "<<filename<<" for reading \nExiting...";
       exit(0);
     }
     size_t read_status = 0;
     while( read_status = fread(&parameter_records, sizeof(config::algorithm), number_of_records, infile) )
     {
        
       
        if(read_status != number_of_records)
        {
          cout<<"\nError in reading from "<<filename;
          printf("\nCould read only %ld of %ld records \nExiting...",read_status,number_of_records);
          exit(0);
        }
        else
        {
          for(int i = 0; i < number_of_records; ++i)
         { 
           printf("\n\nRECORD %d",i);
           printf("\nx_init = %f",parameter_records[i].slmm_parameters.x_init);
           printf("\ny_init = %f",parameter_records[i].slmm_parameters.y_init);
           printf("\nalpha = %f",parameter_records[i].slmm_parameters.alpha);
           printf("\nbeta = %f",parameter_records[i].slmm_parameters.beta);
           printf("\nrotation_rounds = %d",parameter_records[i].rotation_rounds);
           printf("\nswap_rounds = %d",parameter_records[i].swap_rounds);
           printf("\nseed_lut_gen_1 = %u",parameter_records[i].seed_lut_gen_1);
           printf("\nseed_lut_gen_2 = %u",parameter_records[i].seed_lut_gen_2);
           printf("\nseed_row_rotate = %u",parameter_records[i].seed_row_rotate);
           printf("\nseed_col_rotate = %u",parameter_records[i].seed_col_rotate);

         }        
        }
       
       
     }
       
   }
   
   static inline void generateParameterRecords(config::algorithm *&parameter_records,size_t number_of_records)
   {
     cout<<"\nIn generateParameterRecords";
     for(int i = 0; i < number_of_records; ++i)
     {
       parameter_records[i].slmm_parameters.x_init = getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
       parameter_records[i].slmm_parameters.y_init = getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
       parameter_records[i].slmm_parameters.alpha =  getRandomDouble(ALPHA_LOWER_LIMIT,ALPHA_UPPER_LIMIT);
       parameter_records[i].slmm_parameters.beta =   getRandomDouble(double_lower_limit,double_upper_limit);
       parameter_records[i].rotate_rounds =          getRandomUnsignedInteger8(ROTATE_ROUND_LOWER_LOWER_LIMIT,ROTATE_ROUND_UPPER_LIMIT);
       parameter_records[i].swap_rounds =            getRandomUnsignedInteger8(SWAP_ROUND_LOWER_LIMIT,SWAP_ROUND_UPPER_LIMIT);
       parameter_records[i].diffusion_rounds =       getRandomUnsignedInteger8(DIFFUSION_ROUND_LOWER_LIMIT,DIFFUSION_ROUND_UPPER_LIMIT);
       parameters_records[i].seed_lut_gen_1 =        getRandomUnsignedInteger32();
        
     }
   }*/  
   
}

#endif
