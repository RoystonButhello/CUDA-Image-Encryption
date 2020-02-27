#include <iostream> /*For IO*/
#include <cstdint>  /*For stadard variable support*/
#include "functions.hpp"
#include "kernel.hpp"

using namespace cv;
using namespace std;

int main()
{
  cv::Mat image,fractal;
  image=imread("airplane_encrypted.png",IMREAD_COLOR);
  fractal=imread("Gradient.png",IMREAD_COLOR);  

  uint16_t m=0,n=0,mid=0,total=0;
  std::string type=std::string("");
  
  
  cv::resize(image,image,cv::Size(4,4));
  cv::resize(fractal,fractal,cv::Size(4,4));  
  
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
  uint32_t *img_shuffle =(uint32_t*)malloc(sizeof(uint32_t)*total);
  
  cout<<"\nAfter CPU Declarations";
  /*GPU vectors*/
  uint8_t *gpuimgIn;
  uint8_t *gpuimgOut;
  uint8_t *gpuFrac;
  uint32_t *gpuShuffIn;
  uint32_t *gpuShuffOut;
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

  /*FRACTAL XORING*/
  cudaMallocManaged((void**)&gpuFrac,total*3*sizeof(uint8_t));
  
  flattenImage(fractal,fractal_vec);
  
  for(uint32_t i=0;i<total*3;++i)
  {
    gpuFrac[i]=fractal_vec[i];
  }

  dim3 grid_frac_xor(m*n,1,1);
  dim3 block_frac_xor(3,1,1);
  
  run_FracXor(gpuimgIn,gpuimgOut,gpuFrac,grid_frac_xor,block_frac_xor);
  
  for(uint32_t i=0;i<total*3;++i)
  {
    temp=gpuimgIn[i];
    gpuimgIn[i]=gpuimgOut[i];
    gpuimgOut[i]=temp;
  }
  
  for(uint32_t i=0;i<total*3;++i)
  {
    img_vec[i]=gpuimgOut[i];
  }

  cout<<"\nAfter 3 rounds of Dec_GenCatMap, 3 rounds of shuffle, one round of fractal xor and one round of shuffle, img_vec=";
  for(uint32_t i=0;i<total*3;++i)
  {
    printf("%d ",img_vec[i]);
  }

  /*MAPPING IMAGE TO ARNOLD MAP TABLE*/
  cudaMallocManaged((void**)&gpuShuffIn,total*sizeof(uint32_t));
  cudaMallocManaged((void**)&gpuShuffOut,total*sizeof(uint32_t));

  for(uint32_t i=0;i<total;++i)
  {
    img_shuffle[i]=i;
  }
  
  for(uint32_t i=0;i<total;++i)
  {
    gpuShuffIn[i]=img_shuffle[i];
  }
  
  dim3 grid_ar_map_table(m,n,1);
  dim3 block_ar_map_table(1,1,1);
  
  for(uint32_t i=0;i<11;++i)
  {
    run_ArMapTable(gpuShuffIn,gpuShuffOut,grid_ar_map_table,block_ar_map_table);
    for(uint32_t i=0;i<total;++i)
    {
      temp=gpuShuffIn[i];
      gpuShuffIn[i]=gpuShuffOut[i];
      gpuShuffOut[i]=temp;
    }
  }
  
  for(uint32_t i=0;i<total;++i)
  {
    img_shuffle[i]=gpuShuffIn[i];
  }
  
  cout<<"\nAfter 3 rounds of DecGenCat Map, 3 rounds of shuffle, one round of fractal xor, one round of shuffle, 3 rounds of ArMapTable, 3 rounds of Shuffle=,img_shuffle=";

 for(uint32_t i=0;i<total;++i)
 {
   printf("%d ",img_shuffle[i]);
 }

  
  
  return 0;
}

