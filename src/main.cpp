  #include <iostream> /*For IO*/
  #include <cstdint>  /*For standardized data types*/
  #include "functions.hpp"
  #include "kernel.hpp"
  using namespace std;
  using namespace cv;
  int main()
  { 
    
    uint16_t m=0,n=0,total=0;
    uint8_t number_cat_map_rounds=0,temp=0;
    
    cv::Mat image;
    
    /*LOAD AND SQUARE IMAGE. GET CATMAP ROUNDS*/
    image=cv::imread("lena.png",cv::IMREAD_COLOR);
    
    if(RESIZE_TO_DEBUG==1)
    {
      cv::resize(image,image,cv::Size(2,2));
    }
    
    //getSquareImage(image,"original_dimensions.txt",RESIZE_TO_MAXIMUM);
    
    m=(uint16_t)image.rows;
    n=(uint16_t)image.cols;
    
    cout<<"\nm= "<<m<<"\tn= "<<n;
    total=(m*n);
    bool isNotDecimal=0;

    number_cat_map_rounds=getCatMapRounds(CATMAP_ROUND_LOWER,CATMAP_ROUND_UPPER,SEED1);
    
    if (DEBUG_CONSTANTS==1)
    {
      cout<<"\nNumber of Cat Map rounds="<<number_cat_map_rounds;
    }
    
    
    
    /*Declarations*/
    
    /*CPU vector declarations and allocations*/
    std::vector<float> P1(total);
    std::vector<float> P2(total);
    
    uint8_t *img_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    uint8_t *img_empty=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    uint8_t *U=(uint8_t*)malloc(sizeof(uint8_t)*m);
    uint8_t *V=(uint8_t*)malloc(sizeof(uint8_t)*m);
    
    for(uint16_t i=0;i<m;++i)
    {
      U[i]=0;
      V[i]=0;
    }    
    
    for(uint16_t i=0;i<total*3;++i)
    {
      img_vec[i]=0;
      img_empty[i]=0;
    } 
    
    /*GPU vector declarations*/
    uint8_t *gpuimgIn;
    uint8_t *gpuimgOut;
    uint8_t *gpuU;
    uint8_t *gpuV;
     
    /*FLATTEN IMAGE*/
    flattenImage(image,img_vec);  
   
    /*GENERATE RELOCATION VECTORS*/
    genRelocVec(U,P1,"constants1.txt",m,n,SEED1);
    genRelocVec(V,P2,"constants2.txt",n,m,SEED2);
    
    /*Checking P1,P2,U and V*/
    if (DEBUG_VECTORS==1)
    {
      cout<<"\nP1=";
      printFloatVector(P1);
    
      cout<<"\nU=";
      for(uint32_t i=0;i<m;++i)
      {
        printf("%d ",U[i]);
      }
    
      cout<<"\n";
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
    
   if(PRINT_IMAGES==1) 
   {
     cout<<"\nimg_vec=";
     for(uint32_t i=0;i<total*3;++i)
     {
        printf("%d ",img_vec[i]);
     }
   }
    
    /*ARNOLD IMAGE MAPPING*/
    cudaMalloc((void**)&gpuimgIn,total*3*sizeof(uint8_t));
    cudaMalloc((void**)&gpuimgOut,total*3*sizeof(uint8_t));
    
    cudaMemcpy(gpuimgIn,img_vec,total*3*sizeof(uint8_t),cudaMemcpyHostToDevice);
    cudaMemcpy(gpuimgOut,img_empty,total*3*sizeof(uint8_t),cudaMemcpyHostToDevice);
    
    dim3 grid_ar_map_img(m,n,1);
    dim3 block_ar_map_img(3,1,1);
    
    run_ArMapImg(gpuimgIn,gpuimgOut,grid_ar_map_img,block_ar_map_img);
    cudaMemcpy(img_vec,gpuimgOut,4*3*sizeof(uint8_t),cudaMemcpyDeviceToHost);
     
   
   //swapImgVectors(gpuimgIn,gpuimgOut,total*3);
    cout<<"\nimgvec after ArMapImg=";
    for(uint16_t i=0;i<total*3;++i)
    {
       printf("%d ",img_vec[i]);
    }
  
    for(uint8_t i=0;i<total*3;++i)
    {
      temp=img_vec[i];
      img_vec[i]
    }
   
   
     
   return 0; 
  }
  

