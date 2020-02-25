  #include <cstdint>
 
  __global__ void GenCatMap(uint8_t *in, uint8_t *out, uint32_t *colRotate, uint32_t *rowRotate)
  {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int InDex    = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int OutDex   = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
  }

  extern "C" void run_GenCatMap(uint8_t *in, uint8_t *out, uint32_t *colRotate, uint32_t *rowRotate,dim3 blocks,dim3 block_size)
  {
    GenCatMap<<<blocks,block_size>>>(in,out,colRotate,rowRotate);
    
  }
  
