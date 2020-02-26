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
  
