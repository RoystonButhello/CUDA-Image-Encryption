  #include <iostream>
  #include <cstdio>
  #include <cstdint>
  using namespace std;
  

    __global__ void WarmUp()
    {
      return;
    }
    
    
    __global__ void Enc_GenCatMap(uint8_t* in, uint8_t* out, const uint32_t* __restrict__ colRotate, const uint32_t* __restrict__ rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int InDex    = ((gridDim.y)*blockIdx.x + blockIdx.y) * blockDim.x  + threadIdx.x;
        int OutDex   = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * blockDim.x  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }

    __global__ void Dec_GenCatMap(uint8_t* in, uint8_t* out, const uint32_t* __restrict__ colRotate, const uint32_t* __restrict__ rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int OutDex   = ((gridDim.y)*blockIdx.x + blockIdx.y) * blockDim.x  + threadIdx.x;
        int InDex    = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * blockDim.x  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }
   
    
   __global__ void generateU(double* P,uint16_t* U,double n)
   {
       int tid = blockIdx.x *blockDim.x + threadIdx.x;
       U[tid]  = (int)fmod((P[tid]*100000000.00),n);
   }

   __global__ void grayLevelTransform(uint8_t* img_vec, const uint32_t* __restrict__ random_array)
   {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     img_vec[tid] = img_vec[tid] ^ random_array[tid];
   }
   
   __global__ void encRowColSwap(uint8_t* img_in,uint8_t* img_out, const uint32_t* __restrict__ rowSwapLUT, const uint32_t* __restrict__ colSwapLUT)
   {
      int blockId = blockIdx.y * gridDim.x + blockIdx.x;
      int threadId = blockId * blockDim.x + threadIdx.x;
      
      int gray_level_index_in = threadId;
      int row = rowSwapLUT[blockIdx.x];
      int col = colSwapLUT[blockIdx.y];
      int pixel_index_out = row * gridDim.y + col;
      int gray_level_index_out = pixel_index_out * blockDim.x + threadIdx.x;
      img_out[gray_level_index_in] = img_in[gray_level_index_out];
      
   }
   
  __global__ void decRowColSwap(uint8_t* img_in,uint8_t* img_out, const uint32_t* __restrict__ rowSwapLUT,const uint32_t* __restrict__ colSwapLUT)
  {
    int blockId= blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    
    int gray_level_index_in = threadId;
    int row = rowSwapLUT[blockIdx.x];
    int col = colSwapLUT[blockIdx.y];
    int pixel_index_out = row * gridDim.y + col;
    int gray_level_index_out = pixel_index_out * blockDim.x + threadIdx.x;
    img_out[gray_level_index_out] = img_in[gray_level_index_in];
  }   

  extern "C" void run_WarmUp(dim3 blocks,dim3 block_size)
  {
     WarmUp<<<blocks,block_size>>>();
     
  }
  
  extern "C" void run_EncGenCatMap(uint8_t* in,uint8_t* out,const uint32_t* __restrict__ colRotate, const uint32_t* __restrict__ rowRotate,dim3 blocks,dim3 block_size)
  {
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    Enc_GenCatMap<<<blocks,block_size>>>(in,out,colRotate,rowRotate);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("\nTime to rotate:  %3.6f \u03BCs \n", time * 1000.0);
    
  }
  
  extern "C" void run_DecGenCatMap(uint8_t* in,uint8_t* out,const uint32_t* __restrict__ colRotate, const uint32_t* __restrict__ rowRotate,dim3 blocks,dim3 block_size)
  {
     float time;
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start, 0);
     
     Dec_GenCatMap<<<blocks,block_size>>>(in,out,colRotate,rowRotate);
     
     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&time, start, stop);
     
     printf("\nTime to unrotate:  %3.6f \u03BCs \n", time * 1000.0);
        
  }

  
  extern "C" void run_generateU(double* P,uint16_t* U,double n,dim3 blocks,dim3 block_size)
  {
    generateU<<<blocks,block_size>>>(P,U,n);
    
  }

  extern "C" void run_grayLevelTransform(uint8_t* img_vec, const uint32_t* __restrict__ random_array, dim3 blocks, dim3 block_size)
  {
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    grayLevelTransform<<<blocks, block_size>>>(img_vec, random_array);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("\nTime to diffuse:  %3.6f \u03BCs \n", time * 1000.0);
    
  }
  
  extern "C" void run_encRowColSwap(uint8_t* img_in,uint8_t* img_out, const uint32_t* __restrict__ rowSwapLUT, const uint32_t* __restrict__ colSwapLUT,dim3 blocks,dim3 block_size)
  {
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    encRowColSwap<<<blocks, block_size>>>(img_in,img_out,rowSwapLUT,colSwapLUT);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("\nTime to swap:  %3.6f \u03BCs \n", time * 1000.0);
    
  }
  
  extern "C" void run_decRowColSwap(uint8_t* img_in,uint8_t* img_out,const uint32_t* __restrict__ rowSwapLUT,const uint32_t* __restrict__ colSwapLUT,dim3 blocks,dim3 block_size)
  {
     float time;
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start, 0);
    
     decRowColSwap<<<blocks,block_size>>>(img_in,img_out,rowSwapLUT,colSwapLUT);
     
     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&time, start, stop);
     
     printf("\nTime to unswap:  %3.6f \u03BCs \n", time * 1000.0);
  }

