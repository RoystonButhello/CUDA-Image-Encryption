/*These two lines prevent the compiler from reading the same header file twice*/
/**
 * This header file contains functions common to all files in the implementation 
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream> /*For IO*/
#include <cstdio>   /*For printf()*/
#include <string>   /*for std::string*/
#include <random>   /*For Mersenne Twister PRNG*/
#include <chrono>   /*For measuring execution time using std::chrono::system_clock::now()*/
#include <fstream>  /*For file handling*/
#include <cstdint>  /*For standardized variable types*/
#include <cstdbool> /*For boolean variables*/
#include <opencv2/opencv.hpp> /*For opencv*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo> 
#include <cmath>    /*For the standard C Math library*/ 
#include <cstdlib>  /*For malloc() and calloc()*/
#include <ctime>    /*For std::clock()*/
#include <openssl/sha.h> /*For generating SHA256 Hashes*/
#include <iomanip> /*For setw()*/
#include <sstream> /*For stringstream*/
#include <vector>  /*For std::vector*/
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
  static inline uint32_t get_n_mantissa_bits_safe(double f,int number_of_bits);
  

  static inline void writeVectorToFile32(uint32_t *&vec,int length,std::string filename);
  static inline void writeVectorToFile8(uint8_t *&vec,int length,std::string filename);
  
  static inline void printArray8(uint8_t *&arr,int length);
  static inline void printArray16(uint16_t *&arr,int length);
  static inline void printArray32(uint32_t *&arr,int length);
  static inline void printArrayDouble(double *&arr,int length);
  
  static inline uint32_t getRandomUnsignedInteger32(uint32_t lower_bound,uint32_t upper_bound);
  static inline int getRandomInteger(int lower_bound,int upper_bound);
  static inline uint8_t getRandomUnsignedInteger8(uint8_t lower_bound,uint8_t upper_bound);
  static inline double getRandomDouble(double lower_limit,double upper_limit);
  
  static inline config::ChaoticMap mapAssigner(int lower_limit, int upper_limit);
  static inline void rowColLUTGen(uint32_t *&rowSwapLUT,uint32_t *&rowRandVec,uint32_t *&colSwapLUT,uint32_t *&colRandVec,uint32_t m,uint32_t n);
  static inline void swapLUT(uint32_t *&swapLUT,uint32_t *randVec,uint32_t m);
  static inline void genLUTVec(uint32_t *&lutVec,uint32_t n);
  
  static inline void genMapLUTVec(uint32_t *&lut_vec,uint32_t n);
  static inline std::string getFileNameFromPath(std::string filename);
  
  static inline std::string getFileNameFromPath(std::string filename);
  static inline std::string sha256_hash_string (unsigned char hash[SHA256_DIGEST_LENGTH]);
  static inline std::string calc_sha256(const char* path);
  
  static inline void checkImageVectors(uint8_t *plain_img_vec,uint8_t *decrypted_img_vec,uint32_t total);
  static inline bool checkImages(cv::Mat image_1,cv::Mat image_2);
  
  
  /**
   * Converts an image of dimensions N x M into a 1D vector of length N x M. Takes a 2D N X M image, a 1D vector of length N X M, and the number of channels as arguments
   */
  static inline void flattenImage(cv::Mat image,uint8_t *&img_vec,uint32_t channels)
  {
    
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

  /**
   * Prints the gray level values in a cv::Mat image in row major order. Takes 2D N X M image and number of channels as parameters
   */
  static inline void printImageContents(cv::Mat image,uint32_t channels)
  {
    for(uint32_t i=0;i<image.rows;++i)
    { 
      printf("\n");
      for(uint32_t j=0;j<image.cols;++j)
      {
         for(uint32_t k=0;k < channels;++k)
         {
          
          printf("%d\t",image.at<Vec3b>(i,j)[k]); 
         } 
       }
    }
  }

  /**
   * Checks if the product of 2 16-bit unsigned integers exceeds 255. Takes the 2 16-bit unsigned integers as arguments
   */
  static inline uint8_t checkOverflow(uint16_t  number_1,uint16_t number_2)
  {
    
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

  /**
   * formatted output of ieee-754 representation of double-precision floating-point 
   */
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

  
  /**
   * Print bits of a 32-bit signed integer. Takes the number of bits as argument
   */
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

  /**
   * Transfers the last n bits from a double to an n-bit unsigned integer. Takes the double and and the number of bits as arguments
   */
  static inline uint32_t get_n_mantissa_bits_safe(double f,int number_of_bits)
  {
    union {
        double f;
        uint32_t u;
    } fu = { .f = f };
    
    int i=number_of_bits;
    uint8_t bit_store_8=0;
    uint16_t bit_store_16=0;
    uint32_t bit_store_32 = 0;
    
    while (i--)
    {
        
        if(BIT_RETURN(fu.u,i)==1)
        {
            bit_store_32 |= 1 << i;
        }
        
    }
    
    return bit_store_32;
  }

  /**
   * Writes a 32-bit vector to a .txt file. Takes a vector of length 'length', and its length as arguments
   */
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

  /**
   * Writes an 8-bit unsigned integer vector to a .txt file. Takes a vector of length 'length' and file path as arguments
   */
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
  
  /**
   * Prints an 8-bit unsigned integer array of length 'length'. Takes the array and its length as arguments
   */
  static inline void printArray8(uint8_t *&arr,int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf(" %d",arr[i]);
    }
  }

  /**
   * Prints a 16-bit unsigned integer array of length 'length'. Takes the array and its length as arguments
   */
  static inline void printArray16(uint16_t *&arr, int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf(" %d",arr[i]);
    }
  }

  /**
   * Prints a 32-bit unsigned integer array of length 'length'. Takes the array and its length as arguments
   */
  static inline void printArray32(uint32_t *&arr, int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf(" %d",arr[i]);
    }
  }
  
  /**
   * Prints a double array of length 'length'. Takes the array and its length as arguments
   */
  static inline void printArrayDouble(double *&arr,int length)
  {
    for(int i = 0; i < length; ++i)
    {
      printf(" %f",arr[i]);
    }
  }
  
  
  /**
   * Returns a random 8-bit unsigned integer within a range of (lower_bound,upper_bound). Takes the lower_bound and upper_bound as arguments
   */
  static inline uint8_t getRandomUnsignedInteger8(uint8_t lower_bound,uint8_t upper_bound)
  {
      std::random_device r;
      std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
      mt19937 seeder(seed);
      uniform_int_distribution<uint8_t> intGen(lower_bound, upper_bound);
      uint8_t alpha=intGen(seeder);
      return alpha;
  }

  
  /**
   * Returns a random 32-bit unsigned integer within a range of (lower_bound,upper_bound). Takes the lower_bound and upper_bound as arguments
   */
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
  
  
  /**
   * Returns a random 32-bit signed integer within a range of (lower_bound,upper_bound). Takes the lower_bound and upper_bound as arguments
   */
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

  
  /**
   * Returns a random double within a range of (lower_bound,upper_bound). Takes the lower_bound and upper_bound as arguments
   */
  static inline double getRandomDouble(double lower_limit,double upper_limit)
  {
     std::random_device r;
     std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
     mt19937 seeder(seed);
     uniform_real_distribution<double> realGen(lower_limit, upper_limit);   
     auto randnum=realGen(seeder);
     return randnum;
  }
  
  
  /**
   * Returns a value of type ChaoticMap within a range of (lower_limit,upper_limit). Takes the lower_limit and upper_limit as arguments
   */
  static inline config::ChaoticMap mapAssigner(int lower_limit, int upper_limit)
  {
    config::ChaoticMap chaotic_map;
    chaotic_map = (config::ChaoticMap)getRandomInteger(lower_limit,upper_limit);
    return chaotic_map;
  }

  /**
   * Generates shuffled row and column Lookup Tables using Fisher - Yates Shuffle for row and column rotation or swapping. Takes two 1D vectors of length N X M and two vectors of length M-1 and N-1
   */
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
  
  /**
   * Shuffles the Lookup Table used to shuffle chaotic map choices array. Takes a 1D vector of length M-1 and a 1D vector of length N X M and M as arguments
   */
  static inline void swapLUT(uint32_t *&swapLUT,uint32_t *randVec,uint32_t m)
  {

    int jLUT=0;
    for(int i = m - 1; i > 0; i--)
    {
      jLUT = randVec[i] % i;
      std::swap(swapLUT[i],swapLUT[jLUT]);
    }
  
  }
  
  /**
   * Generates a Lookup Table with values from 0 to n - 1 in ascending order for row and column swapping or rotating. Takes a vector of length N and its length as arguments
   */
  static inline void genLUTVec(uint32_t *&lut_vec,uint32_t n)
  {
    for(int i = 0; i < n; ++i)
    {
      lut_vec[i] = i;
    }
  }
  
  /**
   * Generates a Lookup Table with values from 1 to n in ascending order for shuffling the chaotic map choices array. Takes a vector of length N and its length as arguments 
   */
  static inline void genMapLUTVec(uint32_t *&lut_vec,uint32_t n)
  {
    int i = 0;
    for(i = 0; i < n; ++i)
    {
      lut_vec[i] = i + 1;
    }
  }
  
  
  /**
   * Gets the file name from the given file path. Takes the file path as an argument
   */
  static inline std::string getFileNameFromPath(std::string filename)
  {
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
      filename.erase(0, last_slash_idx + 1);
    }

    // Remove extension if present.
    const size_t period_idx = filename.rfind('.');
    if (std::string::npos != period_idx)
    {
      filename.erase(period_idx);
    }
      
    return filename;
  }
  
  
  /**
   * Converts an 8-bit unsigned char SHA256 array into a SHA256 std::string. Takes the 8-bit unsigned char hash array of length 64 as an argument
   */
  static inline std::string sha256_hash_string (unsigned char hash[SHA256_DIGEST_LENGTH])
  {
    

    stringstream ss;
    for(int i = 0; i < SHA256_DIGEST_LENGTH; i++)
    {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    
    return ss.str();

    
  }
  
  /**
   * Calculates SHA256 Hash of a given file. Takes the file path as an argument
   */
  static inline std::string calc_sha256(const char* path)
  {
    FILE* file = fopen(path,"rb");
    
    if(file==NULL) 
    {
        printf("\n File Not found.\n Exiting..."); 
        exit(0);
    }
        
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    const int bufSize = 32768;
    char* buffer = (char*)malloc(bufSize);
    int bytesRead = 0;
    
    if(buffer==NULL) 
    {
        printf("\n File Not found.\n Exiting..."); 
        exit(0);   
    }
    
    while((bytesRead = fread(buffer, 1, bufSize, file)))
    {
        SHA256_Update(&sha256, buffer, bytesRead);
    }
    
    SHA256_Final(hash, &sha256);

    std::string hash_final = sha256_hash_string(hash);
    cout<<"\nSHA256 hash of "<<path<<" is "<<hash_final;
    
    fclose(file);
    return hash_final;
  }      
  
  /**
   * Finds differences between two image vectors. Takes two image vectors of length total and the length total as arguments
   */
  static inline void checkImageVectors(uint8_t *plain_img_vec,uint8_t *decrypted_img_vec,uint32_t total)
  {
    int cnt=0;
    for(int i=0; i < total; ++i)
    {
      if(decrypted_img_vec[i]-plain_img_vec[i]!=0)
      {
        ++cnt;
      }
    
    }
    printf("\nNumber of vector differences= %d",cnt);
    
  }
  
  /**
   * Finds differences between two 2D N x M images. Takes two 2D N X M images as arguments
   */
  static inline bool checkImages(cv::Mat image_1,cv::Mat image_2)
  {
    if(image_1.rows != image_2.rows or image_1.cols != image_2.cols)
    {
      cout<<"\nCould not comapare images\nExiting...";
      exit(0);
    }
    
    uint8_t difference = 0;
    uint32_t count_differences = 0;
    for(int i = 0; i < image_1.rows; ++i)
    {
      for(int j = 0; j < image_1.cols; ++j)
      {
        for(int k = 0; k < image_1.channels(); ++k)
        {
          difference = image_1.at<Vec3b>(i,j)[k] - image_2.at<Vec3b>(i,j)[k];
          if(difference != 0)
          {
            ++count_differences;
          }
        }
      }
    }
    
    if(count_differences != 0)
    {
      cout<<"\nDifferences between decrypted image and plain image = "<<count_differences;
      return 0;
    }
    
    else
    {
      cout<<"\nDifferences between decrypted image and plain image = "<<count_differences;
      return 1;
    }
  }
   
}

#endif

