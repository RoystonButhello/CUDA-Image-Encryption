#include <iostream> /*For IO*/
#include <cstdio>   /*For printf*/
#include <string>   /*For std::string()*/
#include <random>   /*For random number generation*/
#include <chrono>   /*For time*/
#include <fstream>  /*For writing to file*/ 
#include <cstdint>  /*For standard variable types*/
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <cmath>    /*For sqrt()*/ 

using namespace cv;
using namespace std;

/*Function Prototypes*/
uint32_t getLast8Bits(uint32_t number);
uint64_t getManipulatedSystemTime();
uint32_t getLargestPrimeFactor(uint8_t n); 
void generatePRNG(std::vector<uint8_t> &random_array);
uint32_t getSeed(uint8_t lower_bound,uint8_t upper_bound);
void flattenImage(cv::Mat image,std::vector<uint8_t> &img_vec);
void printImageContents(cv::Mat image);
void xorImageVector(std::vector<uint8_t> &img_vec);
void printVectorCircular(std::vector <uint8_t> &img_vec,uint16_t xor_position,uint16_t total);
void xorImage(std::vector<uint8_t> &img_vec,uint16_t total,uint16_t xor_position);

uint32_t getLargestPrimeFactor(uint32_t number)
{
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

uint64_t getManipulatedSystemTime()
{
  uint64_t microseconds_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  uint64_t manip_sys_time=(microseconds_since_epoch%255);  
  //printf("\n\n\nMicroseconds since epoch=%ld\t",microseconds_since_epoch);
  return manip_sys_time;
}


uint32_t getSeed(uint8_t lower_bound,uint8_t upper_bound)
{
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    mt19937 seeder(seed);
    uniform_int_distribution<int> intGen(lower_bound, upper_bound);
    uint32_t alpha=intGen(seeder);
    return alpha;
}

uint32_t getLast8Bits(uint32_t number)
{
  uint32_t  result=number & 0xFF;
  return result;

}

void generatePRNG(std::vector<uint8_t> &random_array)
{
  uint32_t random_number=0,alpha=0,largest_prime_factor=0;
  uint32_t manip_sys_time=0;

  for(uint32_t i=0;i<256;++i)
  {
    alpha=getSeed(1,32);
    manip_sys_time=(uint32_t)getManipulatedSystemTime();
    random_number=manip_sys_time+alpha;
    largest_prime_factor=getLargestPrimeFactor(manip_sys_time);
    
    /*printf("\n\nlargest_prime_factor = %d",largest_prime_factor);
    printf("\n\nmanip_sys_time = %d",manip_sys_time);
    printf("\n\nalpha = %d",alpha);
    printf("\n\nrandom_number= %d",random_number);*/
    
    random_array[i]=getLast8Bits(largest_prime_factor*manip_sys_time);
    //printf("\n\nrandom_array[%d]= %d",i,random_array[i]);
  }
}



void flattenImage(cv::Mat image,std::vector<uint8_t> &img_vec)
{
  int16_t m=0,n=0,total=0;
  m=(uint16_t)image.rows;
  n=(uint16_t)image.cols;
  total=m*n;
  image=image.reshape(1,1);
  for(int i=0;i<total*3;++i)
  {
    img_vec[i]=image.at<uint8_t>(i);
  }
}

void printImageContents(Mat image)
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

void printVectorCircular(std::vector<uint8_t> &img_vec,uint16_t xor_position,uint16_t total)
{
  cout<<"\nCircular Image Vector=";
  for(int i=xor_position;i<xor_position+(total*3);++i)
  {
    printf(" %d",img_vec[i%(total*3)]);
  }
}


void xorImage(std::vector<uint8_t> &img_vec,uint16_t total,uint16_t xor_position)
{ 
    int cnt=0;
    for(int i=xor_position;i<total*3;++i)
    {
      img_vec[i] = img_vec[i] ^ img_vec[i-1];
      cout<<"\n i, i-1"<<"  "<<i<<"  "<<i-1; 
      ++cnt;
    }
      
    cout<<"\ncnt= "<<cnt;
    for(int i=0;i<((total*3)-cnt);++i)
    {
      if(i==0)
      {
        cout<<"\n i, total*3-1"<<"  "<<i<<"  "<<(total*3)-1;
        img_vec[i]=img_vec[i] ^ img_vec[(total*3)-1];
      }
      img_vec[i] = img_vec[i] ^ img_vec[i-1];
      
      if(i!=0)
      {
        cout<<"\n i, i-1"<<"  "<<i<<"  "<<i-1;    
      }
    }
}

int main()
{

    // Read the file and confirm it's been opened
    Mat image= cv::imread("airplane.png", IMREAD_COLOR);
    cv::resize(image,image,cv::Size(2,2));
    if (!image.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    
    uint16_t m=0,n=0,total=0,cnt=0;
    uint32_t alpha=0,tme_8=0;
    uint64_t tme=0;
    uint16_t middle_element=0,xor_position=0;
    
    // Read image dimensions
    m = (uint16_t)image.rows; 
    n = (uint16_t)image.cols;
    total=m*n;
    
    
    uint16_t channels=(uint16_t)image.channels();  
    
    std::vector<uint8_t> random_array(256);
    std::vector<uint8_t> img_vec(m*n*3);    

    cout<<"\nrows= "<<m;
    cout<<"\ncolumns="<<n;
    cout<<"\nchannels="<<channels;
    cout<<"\ntotal="<<total;
    
    printImageContents(image); 
    
    alpha=getSeed(1,32);
 
    generatePRNG(random_array);
    
    /*printf("\nrandom_array=");

    for(uint32_t i=0;i<256;++i)
    {
       printf("%d ",random_array[i]);
    }*/


    flattenImage(image,img_vec);
    
    middle_element=random_array[127];
    xor_position=middle_element%(m*n*channels);
    printf("\nmiddle_element= %d",middle_element);
    printf("\nxor_position= %d",xor_position);    
    
    cout<<"\nImage Vector=";
    for(uint32_t i=0;i<total*3;++i)
    {
      printf("%d ",img_vec[i]);
    }
    //printVectorCircular(img_vec,xor_position,total);
    
    /*Alternate circular idea*/
    xorImage(img_vec,total,xor_position);
    
    cout<<"\nXor'd Image Vector=";
    for(int i=0;i<total*3;++i)
    {
      printf(" %d",img_vec[i]);
    }   
 
    return 0;
}


