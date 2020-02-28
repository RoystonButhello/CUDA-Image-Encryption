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
  //fractal=imread("Gradient.png",IMREAD_COLOR);  

  uint16_t m=0,n=0,mid=0,total=0;
  std::string type=std::string("");
  
  /*LOAD AND SQUARE IMAGE. GET CATMAP ROUNDS*/
  mt19937 seeder(time(0));
  uniform_int_distribution<int> intGen(CATMAP_ROUND_LOWER, CATMAP_ROUND_UPPER);
  auto rounds=intGen(seeder);
  
  if(RESIZE_TO_DEBUG==1)
  {
    cv::resize(image,image,cv::Size(100,100));
    //cv::resize(fractal,fractal,cv::Size(249,249));
  } 
  
  m=(uint16_t)image.rows;
  n=(uint16_t)image.cols;
  
  total=(m*n);
  mid=total/2;

  cout<<"\nm= "<<m<<"\nn= "<<n;
  type=type2str(image.type());
  cout<<"\nimage type= "<<type;
  cout<<"\ntotal="<<total;  

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
  uint32_t *gpuShuffle;

   cout<<"\nAfter GPU Declarations";
  /*BASIC OPERATIONS*/
  
  if(DEBUG_VECTORS==1)
  {  cout<<"\nflattened image=";
    for(uint32_t i=0;i<total*3;++i)
    {
      printf("%d ",img_vec[i]);
    }
  
    for(uint32_t i=0;i<m;++i)
    {
      U[i]=0;
      V[i]=0;
    }

    for(uint32_t i=0;i<total;++i)
    {
      P1[i]=0;
      P2[i]=0;
    }
  }  

  /*GENERATE RELOCATION VECTORS*/
  genRelocVecDec(U,P1,m,n,"constants1.txt");
  genRelocVecDec(V,P2,n,m,"constants2.txt");
  
  if (DEBUG_VECTORS==1)
    {
      cout<<"\nP1=";
      printFloatVector(P1);
    
      cout<<"\nU=";
      for(uint32_t i=0;i<m;++i)
      {
        printf("%d ",U[i]);
      }
    
      cout<<"\nP2=";
      printFloatVector(P2);
      cout<<"\nV=";
    
      for(uint32_t i=0;i<m;++i)
      {
        printf("%d ",V[i]);
      }
      
      cout<<"\n";
  }
    
  if(PRINT_IMAGES==1)
  {
    printImageContents(image);
  }
  flattenImage(image,img_vec);
  if(DEBUG_VECTORS==1)
  { cout<<"\nimg_vec before Dec_GenCatMap";
    for(int i=0;i<total*3;++i)
    {
      printf("%d ",img_vec[i]);
    }
  }
  
  
 
  
  /*GPU WARMUP*
  dim3 grid_gpu_warm_up(1,1,1);
  dim3 block_gpu_warm_up(1,1,1);
  
  run_WarmUp(grid_gpu_warm_up,block_gpu_warm_up);*/
   
  
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
  for(uint32_t i=0;i<PERM_ROUNDS;++i)
  {run_DecGenCatMap(gpuimgIn,gpuimgOut,gpuU,gpuV,grid_dec_gen_cat_map,block_dec_gen_cat_map);
    for(uint32_t i=0;i<total*3;++i)
    {
      temp=gpuimgIn[i];
      gpuimgIn[i]=gpuimgOut[i];
      gpuimgOut[i]=temp;
    }     
  }
  
  if(DEBUG_VECTORS==1)
  {
     
     
    for(uint32_t i=0;i<total*3;++i)
    {
      img_vec[i]=gpuimgOut[i];
    }
  
    cout<<"\nimg_vec after Dec_GenCatMap=";
    for(uint32_t i=0;i<total*3;++i)
    {
      printf("%d ",img_vec[i]);
    }
    
     
     std::ofstream file("img_vec_dec.txt");
     std::string image_elements=std::string("");
     if(!file)
     {
       cout<<"Could not create img_vec_dec.txt\nExiting...";
       exit(1);
     }
     
     for(uint32_t i=0;i<total*3;++i)
     {
       image_elements.append(std::to_string(img_vec[i]));
       image_elements.append("\n");
     }
     
     file<<image_elements;
     file.close();
  }
  
  /*FRACTAL XORING
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
  }*/

  /*MAPPING IMAGE TO ARNOLD MAP TABLE
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
  
  for(uint32_t i=0;i<rounds;++i)
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
  
 if(DEBUG_VECTORS==1)
 {
   cout<<"\nAfter ArMapTable=";

   for(uint32_t i=0;i<total;++i)
   {
     printf("%d ",img_shuffle[i]);
   }
 }*/

 /*ARNOLD IMAGE MAP TO TABLE
 cudaMallocManaged((void**)&gpuShuffle,total*sizeof(uint32_t));
 
 for(uint32_t i=0;i<total;++i)
 {
  gpuShuffle[i]=gpuShuffIn[i];
 
 }
 
 dim3 grid_ar_map_table_to_img(m*n,1,1);
 dim3 block_ar_map_table_to_img(3,1,1);
 
 run_ArMapTabletoImg(gpuimgIn,gpuimgOut,gpuShuffle,grid_ar_map_table_to_img,block_ar_map_table_to_img);
 
 for(uint32_t i=0;i<total*3;++i)
 {
   img_vec[i]=gpuimgOut[i];
 }  
 
  if(DEBUG_VECTORS==1)
  { 
    cout<<"\nimg_vec after ArMapTableTImg=";
    for(uint32_t i=0;i<total*3;++i)
    {
      printf("%d ",img_vec[i]);
    }
  }*/

  if(DEBUG_IMAGES==1)
  {
     cv::Mat img_reshape(m,n,CV_8UC3,img_vec);
     cv::imwrite("airplane_decrypted.png",img_reshape);
  }
  
  return 0;
}

