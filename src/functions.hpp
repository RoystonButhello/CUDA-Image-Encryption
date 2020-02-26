  
  #ifndef FUNCTIONS_H /*To ensure no errors if the header file is included twice*/
  #define FUNCTIONS_H
  
  #include <iostream> /*For IO*/
  #include <functional>
  #include <ctime>    /*For time*/
  #include <cstdint>  /*For standardized data types*/
  #include <cstdio>   /*For printf*/
  #include <random>   /*For random number generation*/
  #include <vector>   /*For dynamic arrays*/
  #include <numeric>  /*For std::iota*/
  #include <cmath>    /*For fmod to do x%1 , y%1 */
  #include <array>    /*For static arrays*/
  #include <string>   /*For strings*/
  #include <fstream>  /*For file handling*/
  #include <bitset>   /*For to_string()*/
  #include <opencv2/opencv.hpp> /*For OpenCV*/
  #include <opencv2/highgui/highgui.hpp>
  #include <cuda.h>/*For CUDA*/
  #include <cuda_runtime.h>
 
  /*Constants*/
  #define RAND_UPPER         32   
  #define SEED1              10
  #define SEED2              32
  #define CATMAP_ROUND_LOWER 8
  #define CATMAP_ROUND_UPPER 16
 
  /*Debug Flags*/
  #define RESIZE_TO_DEBUG     1
  #define DEBUG_CONSTANTS     0
  #define DEBUG_VECTORS       1
  #define DEBUG_IMAGES        0
  #define PRINT_IMAGES        1
  #define DEBUG_GPU_VECTORS   1 
  
  /*Modes*/
  #define RESIZE_TO_MAXIMUM   1
  #define RESIZE_TO_MINIMUM   0
  
  using namespace std;
  using namespace cv;

  /*Function prototypes*/
  
  /*Miscellaneous*/
  bool checkDecimal(std::vector<float> arr,uint32_t total); 
  void printImageContents(cv::Mat image);
  void printFloatVector(std::vector<float> vect);
  void printInt8Vector(std::vector<uint8_t> vect);
  
  /*ENCRYPTION*/
  
  /*Phase 1 Basic operations*/
  uint16_t max(uint16_t m,uint16_t n);
  uint16_t min(uint16_t m,uint16_t n);
  void getSquareImage(cv::Mat image,std::string filename,bool mode);
  uint8_t getCatMapRounds(uint8_t lower_bound,uint8_t upper_bound,uint8_t seed); 
  
  /*Phase 2 generate relocation vectors and flatten image*/
  void genRelocVec(uint8_t *&U,std::vector<float> &P,uint16_t m,uint16_t n,uint8_t seed);
  void flattenImage(cv::Mat image,uint8_t *&img_vec);
  
  /*Phase 3 swap gpuimgIn and gpuimgOut */
  void swapImgVectors(uint8_t *&gpuimgIn,uint8_t *&gpuimgOut,uint16_t size);
   
    
  /*Miscellaneous region begins*/
  bool checkDecimal(std::vector<float> arr,uint32_t TOTAL)
  {
    printf("\nIn checkDecimal\n");
    for(uint32_t i=0;i<TOTAL;++i)
    {
      if(arr[i]>=1.0)
      {
        return 1;
      }      
    }
    return 0;
  }
  
 void printFloatVector(std::vector<float> vect)
 {
  for(uint32_t i=0;i<vect.size();++i )
  {
   printf("%f ",vect[i]);
  }
 }
  
 void printInt8Vector(std::vector<uint8_t> vect)
 {
   for(uint32_t i=0;i<vect.size();++i )
  {
   printf("%d ",vect[i]);
  }
 }
  
  void printImageContents(cv::Mat image)
  {
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

/*Miscellaneous region ends*/ 

 /*ENCRYPTION*/
 
 
 /*Phase 1 region begins*/
 uint16_t max(uint16_t m,uint16_t n)
 {
   return (m > n ? m: n); 
 }
  
 uint16_t min(uint16_t m,uint16_t n)
 {
   return (m < n ? m: n); 
 }
 
 void getSquareImage(cv::Mat image,std::string filename,bool mode)
 {
   uint16_t new_size=0;
   uint16_t m=(uint16_t)image.rows;
   uint16_t n=(uint16_t)image.cols;
   std::string dimensions=std::string("");   
   
   dimensions.append(std::to_string(m));
   dimensions.append("\n");
   dimensions.append(std::to_string(n));
   dimensions.append("\n");

   std::ofstream file(filename);
   if(!file)
   {
     cout<<"\nCould not create "<<filename<<"\nExiting...";
     exit(0);
   }
   
   file<<dimensions;
   file.close();
        
   if(m!=n)
   {
     if(mode==RESIZE_TO_MAXIMUM)
     {
       new_size=max(m,n);
       cv::resize(image,image,cv::Size(new_size,new_size));
     }
     
     else
     {
       new_size=min(m,n);
       cv::resize(image,image,cv::Size(new_size,new_size));
     }
   }
 } 

  uint8_t getCatMapRounds(uint8_t lower_bound,uint8_t upper_bound,uint8_t seed)
  {
    uint8_t number_of_rounds=0;
    srand(seed);
    number_of_rounds=8+(rand()%16);
    return number_of_rounds;
  }
  /*Phase 1 region ends*/
 
 /*Phase 2 region begins*/
 void genRelocVec(uint8_t *&U,std::vector<float> &P,std::string filename,int16_t m,uint16_t n,uint8_t seed)
 {
   /*Initialize PRNGs*/
   srand(seed);
   double unzero=0.0000001;
   uint16_t total=(m*n);
   uint16_t mid=((total)/2);
   
   if(m%2!=0)
   {
     mid=mid+1;
   }
   
   uint64_t exponent=100000000000000;
   
   std::string parameters=std::string("");
   
   /*Initialize random parameters*/
   uint8_t a=2+(rand()%32);
   uint8_t b=2+(rand()%32);
   uint16_t c=1+(a*b);
   uint8_t  offset=1+(rand()%32);
   double x=0 + rand() / ( RAND_MAX / (0.0001 - 1.0 + 1.0) + 1.0);
   double y=0 + rand() / ( RAND_MAX / (0.0001 - 1.0 + 1.0) + 1.0);
   
   /*Converting parameters to string and writing to file*/
   parameters.append(std::to_string(a));
   parameters.append("\n");
   parameters.append(std::to_string(b));
   parameters.append("\n");
   parameters.append(std::to_string(c));
   parameters.append("\n");
   parameters.append(to_string(offset));
   parameters.append("\n");
   
   std::ofstream file(filename);
   
   if(!file)
   {
     cout<<"\nCouldn't write parametes to file.\nExiting...";
     exit(0);
   }
   
   file<<parameters;
   file.close();
   
   /*Skip offset values*/
   for(uint32_t i=0;i<offset;++i)
   {
     x = fmod((x + a*y),1) + unzero;
     y = fmod((b*x + c*y),1) + unzero;
   
   }
   
   /*generate P vector*/
   for(uint32_t i=0;i<mid;++i)
   {
     x = fmod((x + a*y),1) + unzero;
     y = fmod((b*x + c*y),1) + unzero;
     P[2*i]=x;
     P[2*i+1]=y;
     
   }
   
   /*Generate U Vector*/
   for(uint32_t i=0;i<m;++i)
   {
    U[i]=fmod((P[i]*exponent),n);
   }
   
 }

 void flattenImage(cv::Mat image,uint8_t *&img_vec)
 {
   uint32_t l=0;
   for(uint32_t i=0;i<(image.rows);++i)
   {
     for(uint32_t j=0;j<(image.cols);++j)
     {
        for(uint32_t k=0;k<3;++k)
        {
          img_vec[l]=image.at<Vec3b>(i,j)[k];
          l++;
        }
     }
     
   }
    
 } 
 /*Phase 2 region ends*/
 
 /*Phase 3 region begins */
 void swapImgVectors(uint8_t *&gpuimgIn,uint8_t *&gpuimgOut,uint16_t size)
 {
   uint8_t temp=0; 
   for(uint8_t i=0;i<size;++i)
   {
     temp=gpuimgIn[i];
     gpuimgIn[i]=gpuimgOut[i];
     gpuimgOut[i]=temp;
   }
 }
 /*Phase 3 region ends*/

#endif
