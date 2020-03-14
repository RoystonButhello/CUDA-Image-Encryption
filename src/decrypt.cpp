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

  uint16_t m=0,n=0,mid=0,total=0,temp=0;
  std::string type=std::string("");
  
  /*LOAD AND SQUARE IMAGE. GET CATMAP ROUNDS*/
  //mt19937 seeder(time(0));
  //uniform_int_distribution<int> intGen(CATMAP_ROUND_LOWER, CATMAP_ROUND_UPPER);
  //auto rounds=intGen(seeder);
  
  if(RESIZE_TO_DEBUG==1)
  {
    cv::resize(image,image,cv::Size(50,50));
    cv::resize(fractal,fractal,cv::Size(50,50));
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
  uint8_t *img_vec_out=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
  uint16_t *U=(uint16_t*)malloc(sizeof(uint16_t)*m);
  uint16_t *V=(uint16_t*)malloc(sizeof(uint16_t)*m);
  uint8_t *fractal_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
  
  
  
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
  flattenImage(image,img_vec);
  flattenImage(fractal,fractal_vec);  

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
  
  /*Checking P1,P2,U and V*/
    if (DEBUG_VECTORS==1)
    {
      cout<<"\nP1=";
      for(int i=0;i<total;++i)
      {
        printf(" %F",P1[i]);
      }
    
      cout<<"\nU=";
      for(uint32_t i=0;i<m;++i)
      {
        printf("%d ",U[i]);
      }
    
      cout<<"\nP2=";
      for(int i=0;i<total;++i)
      {
        printf(" %F",P2[i]);
      }
      
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

  if(DEBUG_VECTORS==1)
  { cout<<"\nimg_vec before Dec_GenCatMap";
    for(int i=0;i<total*3;++i)
    {
      printf("%d ",img_vec[i]);
    }
  }
  
  
 
  
  /*GPU WARMUP*/
  dim3 grid_gpu_warm_up(1,1,1);
  dim3 block_gpu_warm_up(1,1,1);
  
  run_WarmUp(grid_gpu_warm_up,block_gpu_warm_up);
  //uint8_t temp=0; 
  
  /*ARNOLD MAP DECRYPTION*/
  
  /*Allocating U,V,img_vec and fractal_vec device memory*/
  cudaMalloc((void**)&gpuimgIn,total*3*sizeof(uint8_t));
  cudaMalloc((void**)&gpuimgOut,total*3*sizeof(uint8_t));
  cudaMalloc((void**)&gpuU,m*sizeof(uint16_t));
  cudaMalloc((void**)&gpuV,m*sizeof(uint16_t));
  cudaMalloc((void**)&gpuFrac,total*3*sizeof(uint8_t)); 
    
  
  /*Transferring U,V,img_vec and fractal_vec from host to device memory*/
  cudaMemcpy(gpuimgIn,img_vec,total*3*sizeof(uint8_t),cudaMemcpyHostToDevice);
  cudaMemcpy(gpuU,U,m*sizeof(uint16_t),cudaMemcpyHostToDevice);
  cudaMemcpy(gpuV,V,m*sizeof(uint16_t),cudaMemcpyHostToDevice);
  cudaMemcpy(gpuFrac,fractal_vec,total*3*sizeof(uint8_t),cudaMemcpyHostToDevice);
   
  dim3 grid_dec_gen_cat_map(m,n,1);
  dim3 block_dec_gen_cat_map(3,1,1);
  
  
  
  
  for(uint32_t i=0;i<2;++i)
  {run_DecGenCatMap(gpuimgIn,gpuimgOut,gpuU,gpuV,grid_dec_gen_cat_map,block_dec_gen_cat_map);
   swap(gpuimgIn,gpuimgOut); 
  }
  
  /*if(DEBUG_VECTORS==1)
  {
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
  }*/
  
  /*FRACTAL XORING*/
  
  dim3 grid_frac_xor(m*n,1,1);
  dim3 block_frac_xor(3,1,1);
  
  run_FracXor(gpuimgOut,gpuimgIn,gpuFrac,grid_frac_xor,block_frac_xor);
  temp=0;
  swap(gpuimgOut,gpuimgIn); 
  
    
  /*Transferring img_vec from device to Host*/
  cudaMemcpy(gpuimgIn,img_vec,total*3*sizeof(uint8_t),cudaMemcpyDeviceToHost);
  cudaMemcpy(gpuimgOut,img_vec_out,total*3*sizeof(uint8_t),cudaMemcpyDeviceToHost);
   
  
  if(DEBUG_VECTORS==1)
  {
    cout<<"\ngpuimgIn in fracxor Decrypt=";
    for(uint32_t i=0;i<total*3;++i)
    {
      printf("%d ",gpuimgIn[i]);
    }

    cout<<"\ngpuimgOut in fracxor Decrypt=";
    for(uint32_t i=0;i<total*3;++i)
    {
      printf("%d ",gpuimgOut[i]);
    }

    cout<<"\nDecrypted img_vec=";
    for(uint32_t i=0;i<total*3;++i)
    {
       printf("%d ",img_vec[i]);
    }
      
  }
  

  if(DEBUG_IMAGES==1)
  {
     cv::Mat img_reshape(m,n,CV_8UC3,img_vec);
     cv::imwrite("airplane_decrypted.png",img_reshape);
  }
  
  return 0;
}

