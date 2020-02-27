#include <iostream> /*For IO*/
#include <cstdint>  /*For stadard variable support*/
#include "functions.hpp"
#include "kernel.hpp"

using namespace cv;
using namespace std;

int main()
{
  cv::Mat image;
  image=imread("airplane_encrypted.png",IMREAD_COLOR);
  
  uint16_t m=0,n=0,mid=0,total=0;
  std::string type=std::string("");
  
  
  cv::resize(image,image,cv::Size(4,4));  
  
  m=(uint16_t)image.rows;
  n=(uint16_t)image.cols;
  
  total=(m*n);
  mid=total/2;

  cout<<"\nm= "<<m<<"\nn= "<<n;
  type=type2str(image.type());
  cout<<"\nimage type= "<<type;  

  /*Declarations*/
    
  /*CPU vector declarations and allocations*/
  double *P1=(double*)malloc(sizeof(double)*total);
  double *P2=(double*)malloc(sizeof(double)*total);
  uint8_t *img_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
  uint8_t *img_empty=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
  uint16_t *U=(uint16_t*)malloc(sizeof(uint16_t)*m);
  uint16_t *V=(uint16_t*)malloc(sizeof(uint16_t)*m);
  uint8_t *fractal_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
  
  cout<<"\nAfter CPU Declarations";
  /*GPU vectors*/
  uint8_t *gpuimgIn;
  uint8_t *gpuimgOut;
  uint8_t *gpuShuffIn;
  uint8_t *gpuShuffOut;
  uint16_t *gpuU;
  uint16_t *gpuV;

   cout<<"\nAfter GPU Declarations";
  /*BASIC OPERATIONS*/
  
  printImageContents(image);
  
  flattenImage(image,img_vec);
  
  
  
  cout<<"\nflattened image=";
  for(uint32_t i=0;i<total*3;++i)
  {
    printf("%d ",img_vec[i]);
  }
  
  for(uint32_t i=0;i<m;++i)
  {
    U[i]=i;
    V[i]=i;
  }

  for(uint32_t i=0;i<total;++i)
  {
    P1[i]=0;
    P2[i]=0;
  }
  
  /*GPU WARMUP*/
  dim3 grid_gpu_warm_up(1,1,1);
  dim3 block_gpu_warm_up(1,1,1);
  
  run_WarmUp(grid_gpu_warm_up,block_gpu_warm_up);
   
  
  /*ARNOLD MAP DECRYPTION*/
  cudaMallocManaged((void**)&gpuimgIn,total*3*sizeof(uint8_t));
  cudaMallocManaged((void**)&gpuimgOut,total*3*sizeof(uint8_t));
  cudaMallocManaged((void**)&gpuU,m*sizeof(uint16_t));
  cudaMallocManaged((void**)&gpuV,m*sizeof(uint16_t));
  
  for(uint32_t i=0;i<total*3;++i)
  {
    gpuimgIn[i]=img_vec[i];
  }
  
  for(uint32_t i=0;i<m;++i)
  {
    gpuU[i]=U[i];
    gpuV[i]=V[i];
  }

  dim3 grid_dec_gen_cat_map(m,n,1);
  dim3 block_dec_gen_cat_map(3,1,1);
  
  
  
  uint8_t temp=0;
  for(uint32_t i=0;i<3;++i)
  {run_DecGenCatMap(gpuimgIn,gpuimgOut,gpuU,gpuV,grid_dec_gen_cat_map,block_dec_gen_cat_map);
    for(uint32_t i=0;i<total*3;++i)
    {
      temp=gpuimgIn[i];
      gpuimgIn[i]=gpuimgOut[i];
      gpuimgOut[i]=temp;
    }     
  }
  
  for(uint32_t i=0;i<total*3;++i)
  {
    img_vec[i]=gpuimgOut[i];
  }
  
  cout<<"\nimg_vec after 3 rounds of Dec_GenCatMap and 3 rounds of shuffling=";
  for(uint32_t i=0;i<total*3;++i)
  {
    printf("%d ",img_vec[i]);
  }  

  return 0;
}

