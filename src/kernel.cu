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
