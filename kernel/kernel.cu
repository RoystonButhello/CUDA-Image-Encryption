  #include <cstdint>
    __global__ void ArMapImg(uint8_t *in, uint8_t *out)
    {
        int nx = (2*blockIdx.x + blockIdx.y) % gridDim.x;
        int ny = (blockIdx.x + blockIdx.y) % gridDim.y;
        int InDex = ((gridDim.x)*blockIdx.y + blockIdx.x) * 3  + threadIdx.x;
        int OutDex = ((gridDim.x)*ny + nx) * 3 + threadIdx.x;
        out[OutDex] = in[InDex];
    }

    __global__ void WarmUp()
    {
      return;
    }
    
    __global__ void FracXor(uint8_t *in, uint8_t *out, uint8_t *fractal)
    {
        int idx = blockIdx.x * 3 + threadIdx.x;
        out[idx] = in[idx]^fractal[idx];
    } 

    __global__ void Enc_GenCatMap(uint8_t *in, uint8_t *out, uint16_t *colRotate, uint16_t *rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int InDex    = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int OutDex   = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }

    __global__ void Dec_GenCatMap(uint8_t *in, uint8_t *out, uint16_t *colRotate, uint16_t *rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int OutDex   = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int InDex    = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }
   
    
    __global__ void ArMapTable(uint32_t *in, uint32_t *out)
    {
        int nx = (2*blockIdx.x + blockIdx.y) % gridDim.x;
        int ny = (blockIdx.x + blockIdx.y) % gridDim.y;
        int InDex = ((gridDim.x)*blockIdx.y + blockIdx.x);
        int OutDex = ((gridDim.x)*ny + nx);
        out[OutDex] = in[InDex];
    }
   
    __global__ void ArMapTabletoImg(uint8_t *in, uint8_t *out, uint32_t *table)
    {
        uint32_t InDex = blockIdx.x * 3 + threadIdx.x;
        uint32_t OutDex = table[blockIdx.x] * 3 + threadIdx.x;
        out[OutDex] = in[InDex];
    }

   __global__ void generateU(double *P,uint16_t *U,double n)
   {
       int tid = blockIdx.x *blockDim.x + threadIdx.x;
       U[tid]  = (int)fmod((P[tid]*100000000.00),n);
   }

   __global__ void grayLevelTransform(uint8_t *img_vec, uint16_t *random_array)
   {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     img_vec[tid] = img_vec[tid] ^ random_array[tid];
   }
   
   __global__ void encRowColSwap(uint8_t *img_in,uint8_t *img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT)
   {
      int blockId = blockIdx.y * gridDim.x + blockIdx.x;
      int threadId = blockId * blockDim.x + threadIdx.x;
      
      int gray_level_index_in = threadId;
      int row = rowSwapLUT[blockIdx.x];
      int col = colSwapLUT[blockIdx.y];
      int pixel_index_out = row * gridDim.x + col;
      int gray_level_index_out = pixel_index_out * blockDim.x + threadIdx.x;
      img_out[gray_level_index_in] = img_in[gray_level_index_out];
      
   }
   
  __global__ void decRowColSwap(uint8_t *img_in,uint8_t *img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT)
  {
    int blockId= blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    
    int gray_level_index_in = threadId;
    int row = rowSwapLUT[blockIdx.x];
    int col = colSwapLUT[blockIdx.y];
    int pixel_index_out = row * gridDim.x + col;
    int gray_level_index_out = pixel_index_out * blockDim.x + threadIdx.x;
    img_out[gray_level_index_out] = img_in[gray_level_index_in];
  }   

   extern "C" void run_ArMapImg(uint8_t *in, uint8_t *out,dim3 blocks,dim3 block_size)
   {
     ArMapImg<<<blocks,block_size>>>(in,out);
     cudaDeviceSynchronize();
   }

   extern "C" void run_WarmUp(dim3 blocks,dim3 block_size)
   {
     WarmUp<<<blocks,block_size>>>();
     cudaDeviceSynchronize();
   }
  
  extern "C" void run_FracXor(uint8_t *in,uint8_t *out,uint8_t *fractal,dim3 blocks,dim3 block_size)
  {
    FracXor<<<blocks,block_size>>>(in,out,fractal);
    cudaDeviceSynchronize();  
  }

  extern "C" void run_EncGenCatMap(uint8_t *in,uint8_t *out,uint16_t *colRotate,uint16_t *rowRotate,dim3 blocks,dim3 block_size)
  {
    Enc_GenCatMap<<<blocks,block_size>>>(in,out,colRotate,rowRotate);
    cudaDeviceSynchronize();
  }
  
  extern "C" void run_DecGenCatMap(uint8_t *in,uint8_t *out,uint16_t *colRotate,uint16_t *rowRotate,dim3 blocks,dim3 block_size)
  {
     Dec_GenCatMap<<<blocks,block_size>>>(in,out,colRotate,rowRotate);
     cudaDeviceSynchronize();    
  }

  extern "C" void run_ArMapTable(uint32_t *in,uint32_t *out,dim3 blocks, dim3 block_size)
  {
    ArMapTable<<<blocks,block_size>>>(in,out);
    cudaDeviceSynchronize();
  }

  extern "C" void run_ArMapTabletoImg(uint8_t *in,uint8_t *out,uint32_t *table,dim3 blocks,dim3 block_size)
  {
    ArMapTabletoImg<<<blocks,block_size>>>(in,out,table);
    cudaDeviceSynchronize();
  }
  
  extern "C" void run_generateU(double *P,uint16_t *U,double n,dim3 blocks,dim3 block_size)
  {
    generateU<<<blocks,block_size>>>(P,U,n);
    cudaDeviceSynchronize();
  }

  extern "C" void run_grayLevelTransform(uint8_t *img_vec, uint16_t *random_array, dim3 blocks, dim3 block_size)
  {
    grayLevelTransform<<<blocks, block_size>>>(img_vec, random_array);
  }
  
  extern "C" void run_encRowColSwap(uint8_t *img_in,uint8_t *img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,dim3 blocks,dim3 block_size)
  {
    encRowColSwap<<<blocks, block_size>>>(img_in,img_out,rowSwapLUT,colSwapLUT);
  }
  
  extern "C" void run_decRowColSwap(uint8_t *img_in,uint8_t *img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,dim3 blocks,dim3 block_size)
  {
    decRowColSwap<<<blocks,block_size>>>(img_in,img_out,rowSwapLUT,colSwapLUT);
  }
