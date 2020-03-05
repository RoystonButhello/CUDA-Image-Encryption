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
       int tid=blockIdx.x *blockDim.x + threadIdx.x;
       U[tid]=(int)fmod((P[tid]*100000000.00),n);
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
