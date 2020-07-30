//CUDA kernels and CUDA kernel-related function definitions

#include <cstdint>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cub-1.8.0/cub/cub.cuh"

using namespace cub;
using namespace std;
using namespace thrust;

// Warm-up Kernel
__global__ void KWarmUp()
{
    return;
}

// ENC::Permutation by Rotation
__global__ void ENC_RotatePerm(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int InDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * blockDim.x + threadIdx.x;
    int OutDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * blockDim.x + threadIdx.x;
    out[OutDex] = in[InDex];
}

// DEC::Permutation by Rotation
__global__ void DEC_RotatePerm(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int InDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * blockDim.x + threadIdx.x;
    int OutDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * blockDim.x + threadIdx.x;
    out[InDex] = in[OutDex];
}

// Diffusion (top-down)
__global__ void DIFF_TD(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xRow, const double* __restrict__ yRow, const int rows, const double alpha, const double beta, const double myu, const double r, const int map)
{
    // Initialize parameters
    double x = xRow[blockIdx.x];
    double y = yRow[blockIdx.x];
    const int stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    double x_bar = 0;
    double y_bar = 0; 
    // Each thread diffuses one channel of a column
    
    //Arnold Map
    if(map == 1)
    {
      //std::printf("\nDIFF_TD KERNEL ARNOLD MAP\n");
      for(int i = 0; i < rows; i++, idx += stride)
      {
        auto xtmp = x + y;
        y = x + 2 * y;
        x = xtmp - (int)xtmp;
        y = y - (int)y;
        out[idx] = in[idx] ^ (uint8_t)(x * 256);
        
      }
    }
      
    //2D Logistic Map
    else if(map == 2)
    {
      //std::printf("\nDIFF_TD KERNEL 2D LOGISTIC MAP\n");
      for (int i = 0; i < rows; i++, idx += stride)
      {
        x = r * (3 * y + 1) * x * (1 - x);
        y = r * (3 * x + 1) * y * (1 - y);
        out[idx] = in[idx] ^ (uint8_t)(x * 256);
      }
    }
    
    //2D Sine Logistic Modulation Map
    else if(map == 3)
    {
      for (int i = 0; i < rows; i++, idx += stride)
      {
        x = alpha * (sin(M_PI * y) + beta) * x * (1 - x);
        y = alpha * (sin(M_PI * x) + beta) * y * (1 - y);
        out[idx] = in[idx] ^ (uint8_t)(x * 256);
      }
    }
    
    //2D Logistic Adjusted Sine Map
    else if(map == 4)
    {
      for (int i = 0; i < rows; i++, idx += stride)
      {
        x = sin(M_PI * myu * (y + 3) * x * (1 - x));
        y = sin(M_PI * myu * (x + 3) * y * (1 - y));
        out[idx] = in[idx] ^ (uint8_t)(x * 256);
      }
    }
    
    //2D Logistic Adjusted Logistic Map
    else if(map == 5)
    {
     
      for (int i = 0; i < rows; i++, idx += stride)
      {
        x_bar = myu * (y * 3) * x * (1 - x);
        x = 4 * x_bar * (1 - x_bar);
        y_bar = myu * (x + 3) * y * (1 - y);
        y = 4 * y_bar * (1 - y_bar);
        out[idx] = in[idx] ^ (uint8_t)(x * 256);
      }  
     
    }
    
    else
    {
      std::printf("\nInvalid chaotic map choice in DIFF_TD kernel");
    }
} 

// ENC::SELF-XOR (left-right)
__global__ void ENC_XOR_LR(uint8_t* __restrict__ in, const int cols)
{
    // Initialize parameters
    int prev = cols * blockIdx.x * blockDim.x + threadIdx.x;
    int curr = prev + blockDim.x;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        in[curr] = in[curr] ^ in[prev];
        prev = curr;
        curr += blockDim.x;
    }
}

// DEC::SELF-XOR (left-right)
__global__ void DEC_XOR_LR(uint8_t* __restrict__ img, const int cols)
{
    // Initialize parameters
    int curr = cols * blockIdx.x * blockDim.x + threadIdx.x + (cols - 1) * blockDim.x;
    int next = curr - blockDim.x;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        img[curr] = img[curr] ^ img[next];
        curr = next;
        next -= blockDim.x;
    }
}

// Wrappers for kernel calls
extern "C" void kernel_WarmUp()
{
    KWarmUp <<<1, 1>>> ();
}

extern "C" void Wrap_RotatePerm(uint8_t * in, uint8_t * out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block, const int mode)
{
    if (mode == 1)
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        ENC_RotatePerm <<<grid, block>>> (in, out, colRotate, rowRotate);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
    
        std::printf("\nTime to permute:  %3.6f ms \n", time);
    }
    
    else
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
           
        DEC_RotatePerm <<<grid, block>>> (in, out, colRotate, rowRotate);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
    
        std::printf("\nTime to unpermute:  %3.6f ms \n", time);
    }
}

extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& randRowX, const double*& randRowY, const int dim[], const double alpha, const double beta, const double myu, const double r, const int mode, const int map)
{
    // Set grid and block size
    const dim3 gridCol(dim[0], 1, 1);
    const dim3 gridRow(dim[1],1, 1);
    const dim3 block(dim[2], 1, 1);
    
    if (mode == 1)
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        DIFF_TD <<<gridRow, block>>> (in, out, randRowX, randRowY, dim[0], alpha, beta, myu, r, map);
        ENC_XOR_LR <<<gridRow, block>>> (out, dim[0]);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
   
        std::printf("\nTime to diffuse:  %3.6f ms \n", time);
    }

    else
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        DEC_XOR_LR <<<gridRow, block>>> (in, dim[0]);
        DIFF_TD <<<gridRow, block>>> (in, out, randRowX, randRowY, dim[0], alpha, beta, myu, r, map);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
    
        std::printf("\nTime to undiffuse:  %3.6f ms \n", time);
    }
}

extern "C" void Wrap_imageSumReduce(uint8_t* __restrict__ image_vec, uint32_t *device_result, const int dim[])
{
  int num_items = dim[0] * dim[1] * dim[2];
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  //Run the reduction function to check how much temporary storage is needed
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, image_vec, device_result, num_items);
  
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  //Run the reduction function to get the sum of the image
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, image_vec, device_result, num_items);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("\nTime to reduce sum:  %3.6f ms \n", time);
}


