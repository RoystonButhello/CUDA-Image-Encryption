  
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

  #define RAND_UPPER 32   
  #define WRITE_ALPHA_BETA 0
  #define WRITE_OTHER_CONSTANTS 1
  #define SEED1 10
  #define SEED2 32
 
  #define RESIZE_TO_DEBUG 1
  #define DEBUG_CONSTANTS 1
  #define DEBUG_VECTORS   1
  #define DEBUG_IMAGES    0
  
  using namespace std;
  using namespace cv;

  /*Function Prototypes*/
  void flattenGrayScaleImage(cv::Mat image,std::vector<uint8_t> &img_vec);
  void printImageContents(cv::Mat image);
  bool checkDecimal(std::vector<float> arr,uint32_t total);
  void genRelocVec(std::vector<uint8_t> &U,std::vector<float> &P,uint16_t m,uint16_t n,uint8_t seed);
  void printFloatVector(std::vector<float> vect);
  void printInt8Vector(std::vector<uint8_t> vect);
    

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
    for(uint8_t i=0;i<image.rows;++i)
    { printf("\n");
      //printf("\ni=%d\t",i);
      for(uint8_t j=0;j<image.cols;++j)
      {
        //printf("\nj=%d\t",j);
        printf("%d\t",image.at<uchar>(i,j)); 
      }
    }
  }  
 
 void flattenGrayScaleImage(cv::Mat image,std::vector<uint8_t> &img_vec)
 {
   ;
    
 }

 
 void genRelocVec(std::vector<uint8_t> &U,std::vector<float> &P,std::string filename,int16_t m,uint16_t n,uint8_t seed)
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
    U[i]=fmod((P[i]*exponent),m);
   }
   
   
 } 
 
#endif
