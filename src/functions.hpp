  
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
  #include <sstream>  /*To convert strings to numbers*/
  #include <opencv2/opencv.hpp> /*For OpenCV*/
  #include <opencv2/highgui/highgui.hpp>
  #include <cuda.h>
  #include <cuda_runtime.h>
 
  /*Constants*/
  #define RAND_UPPER         32   
  #define SEED1              10
  #define SEED2              32
  #define CATMAP_ROUND_LOWER 8
  #define CATMAP_ROUND_UPPER 16
  #define PERMINTLIM         32
  #define PERM_ROUNDS        8
 
  /*Debug Flags*/
  #define RESIZE_TO_DEBUG     1
  #define DEBUG_CONSTANTS     0
  #define DEBUG_VECTORS       0
  #define DEBUG_IMAGES        1
  #define PRINT_IMAGES        0
  #define DEBUG_GPU_VECTORS   0 
  
  /*Modes*/
  #define RESIZE_TO_MAXIMUM   1
  #define RESIZE_TO_MINIMUM   0
  #define WRITE_TO_FILE       1
  #define READ_FROM_FILE      0
  
  using namespace std;
  using namespace cv;

  /*Function prototypes*/
  
  /*Miscellaneous*/
  bool checkDecimal(std::vector<float> arr,uint32_t total); 
  void printImageContents(cv::Mat image);
  void printFloatVector(double *vect);
  void printInt8Vector(std::vector<uint8_t> vect);
  std::string type2str(int type);

  /*ENCRYPTION*/
  
  /*Phase 1 Basic operations*/
  uint16_t max(uint16_t m,uint16_t n);
  uint16_t min(uint16_t m,uint16_t n);
  void getSquareImage(cv::Mat image,std::string filename,bool mode);
  
  /*Phase 2 generate relocation vectors and flatten image*/
  void genRelocVecEnc(uint16_t *&U,double *&P,uint16_t m,uint16_t n,std::string filename);
  void genRelocVecDec(uint16_t *&U,double *&P,uint16_t m,uint16_t n,const char *filename);
  void flattenImage(cv::Mat image,uint8_t *&img_vec);
  
  /*Phase 3 swap gpuimgIn and gpuimgOut */
   
  /*Phase 6 get Fractal*/
  void getFractal(cv::Mat &fractal,uint16_t m,uint16_t n); 



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
  
 void printFloatVector(double *vect)
 {
  for(uint32_t i=0;i<16;++i )
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

std::string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
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

 void genRelocVecEnc(uint16_t *&U,double *&P,uint16_t m,uint16_t n,std::string filename)
 {
   /*Initialize PRNGs*/
    double unzero = 0.0000000001;
    mt19937 seeder(time(0));
    uniform_int_distribution<int> intGen(1, 32);
    uniform_real_distribution<double> realGen(unzero, 1);
    //int a=2,b=15,c=31,offset=32;
    double x=0.934752,y=0.584024;
    uint16_t total=0,mid=0;
    total=(m*n);
    mid=(m*n)/2;
    int exponent = (int)pow(10, 8);
    cout<<"\nmid="<<mid<<"\ntotal="<<total;
    

   if(m%2!=0)
   {
     mid=mid+1;
   }
   
   
   std::string parameters=std::string("");
   
   /*Initialize random parameters*/
    auto a = intGen(seeder);
    auto b = intGen(seeder);
    auto c = a * b + 1;
    auto offset = intGen(seeder);
   
   printf("\nBefore writing a= %d\tb= %d\tc= %d\toffset= %d\tx= %F\ty= %F",a,b,c,offset,x,y);
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
   printf("\nIn skip offset values");
   for(uint32_t i=0;i<offset;++i)
   {
    
     //printf("\n x=%F,a=%d,b=%d,c=%d,y=%F",x,a,b,c,y);
     x = fmod((x + a*y),1) + unzero;
     y = fmod((b*x + c*y),1) + unzero;
   
   }
   
   /*generate P vector*/
   printf("\nIn P vec gen");
   for(uint32_t i=0;i<mid;++i)
   { 
     //printf("\n x=%F,a=%d,b=%d,c=%d,y=%F",x,a,b,c,y);
     x = fmod((x + a*y),1) + unzero;
     y = fmod((b*x + c*y),1) + unzero;
     P[2*i]=x;
     P[2*i+1]=y;
     
   }
   
   /*Generate U Vector*/
   for(uint32_t i=0;i<m;++i)
   {
    U[i]=(int)fmod((P[i]*exponent),n);
   }
   
 }

  
 void genRelocVecDec(uint16_t *&U,double *&P,uint16_t m,uint16_t n,const char *filename)
 {
   /*Initialize PRNGs*/
    double unzero = 0.0000000001;
    uint16_t total=0,mid=0;
    int a=0,b=0,c=0,offset=0;
    double x=0.934752,y=0.584024;
    total=(m*n);
    mid=(m*n)/2;
    int exponent = (int)pow(10, 8);
    
    cout<<"\nmid="<<mid<<"\ntotal="<<total;
   if(m%2!=0)
   {
     mid=mid+1;
   }
   
    FILE *in_file;

    in_file = fopen(filename, "r");

    if (in_file == NULL)
    {
        printf("Can't open file for reading.\n");
    }
    else
    {
        fscanf(in_file, "%d", &a);
        fscanf(in_file, "%d", &b);
        fscanf(in_file, "%d", &c);
        fscanf(in_file, "%d", &offset);
        
        //printf("Read Constants a= %d b= %d c= %d offset= %d x= %F y= %F",a,b,c,offset,x,y);
        fclose(in_file);
    }
    
    
    printf("\nAfter reading a= %d\tb= %d\tc= %d\toffset= %d\tx= %F\ty= %F",a,b,c,offset,x,y);
   
   /*Skip offset values*/
   printf("\nIn skip offset values");
   for(uint32_t i=0;i<offset;++i)
   { 
     //printf("\n x=%F,a=%d,b=%d,c=%d,y=%F",x,a,b,c,y);
     x = fmod((x + a*y),1) + unzero;
     y = fmod((b*x + c*y),1) + unzero;
   
   }
   
   /*generate P vector*/
   printf("\nIn P vec gen");
   for(uint32_t i=0;i<mid;++i)
   {
     //printf("\n x=%F,a=%d,b=%d,c=%d,y=%F",x,a,b,c,y);
     x = fmod((x + a*y),1) + unzero;
     y = fmod((b*x + c*y),1) + unzero;
     P[2*i]=x;
     P[2*i+1]=y;
     
   }
   
   /*Generate U Vector*/
   for(uint32_t i=0;i<m;++i)
   {
    U[i]=(int)fmod((P[i]*exponent),n);
   }
   
 }  
  
  /*Phase 1 region ends*/
 
 
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

 /*Phase 3 region ends*/

 /*Phase 6 region begins*/
 
 void getFractal(cv::Mat &image,uint16_t m,uint16_t n)
 {
   cv::resize(image,image,cv::Size(m,n));

 } 



#endif
