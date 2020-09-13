//CUDA kernels and CUDA kernel-related function definitions

#include <cstdint>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cub-1.8.0/cub/cub.cuh"

using namespace cub;
using namespace std;
using namespace thrust;

/**
 * Warm-up CUDA kernel. Used for warming up GPU for accurate benchmarking. Takes no arguments
 */
__global__ void KWarmUp()
{
    return;
}

/**
 * ENC::Permutation by Rotation. Permutation CUDA kernel. Takes input and output image vectors and permutation vectors as arguments
 */
__global__ void ENC_RotatePerm(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int InDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * blockDim.x + threadIdx.x;
    int OutDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * blockDim.x + threadIdx.x;
    out[OutDex] = in[InDex];
}

/**
 * DEC::Unpermutation by Rotation. Unpermutation CUDA kernel. Takes input and output image vectors and permutation vectors as arguments
 */
__global__ void DEC_RotatePerm(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int InDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * blockDim.x + threadIdx.x;
    int OutDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * blockDim.x + threadIdx.x;
    out[InDex] = in[OutDex];
}

/**
 * Diffusion (top - down). Diffusion CUDA kernel. Takes input and output image vectors and diffusion vectors, 2D Logistic Map control parameter and diffuse propagation factor as arguments
 */
__global__ void DIFF_TD(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xRow, const double* __restrict__ yRow, const int rows, const double r, uint32_t diffuse_propagation_factor)
{
    // Initialize parameters
    double x = xRow[blockIdx.x];
    double y = yRow[blockIdx.x];
    const int stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    //double x_bar = 0;
    //double y_bar = 0; 
    // Each thread diffuses one channel of a column
    
    
    //std::printf("\nDIFF_TD KERNEL 2D LOGISTIC MAP\n");
    for (int i = 0; i < rows; i++, idx += stride)
    {
      x = r * (3 * y + 1) * x * (1 - x);
      y = r * (3 * x + 1) * y * (1 - y);
      out[idx] = in[idx] ^ diffuse_propagation_factor ^ (uint8_t)(x * 256);
    }
    
    
    
} 

/**
 * ENC::SELF-XOR (left-right). Self - xor CUDA kernel. Takes input image vector, diffuse propagation factor and number of image columns as arguments 
 */
__global__ void ENC_XOR_LR(uint8_t* __restrict__ in, uint32_t diffuse_propagation_factor, const int cols)
{
    // Initialize parameters
    int prev = cols * blockIdx.x * blockDim.x + threadIdx.x;
    int curr = prev + blockDim.x;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        in[curr] = in[curr] ^ diffuse_propagation_factor ^ in[prev];
        prev = curr;
        curr += blockDim.x;
    }
}

/** 
 * DEC::SELF-XOR (right-left). Self - xor CUDA kernel. Self - xor CUDA kernel. Takes input image vector, diffuse propagation factor and number of image columns as arguments
 */ 
__global__ void DEC_XOR_LR(uint8_t* __restrict__ img, uint32_t diffuse_propagation_factor, const int cols)
{
    // Initialize parameters
    int curr = cols * blockIdx.x * blockDim.x + threadIdx.x + (cols - 1) * blockDim.x;
    int next = curr - blockDim.x;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        img[curr] = img[curr] ^ diffuse_propagation_factor ^ img[next];
        curr = next;
        next -= blockDim.x;
    }
}

// Wrappers for kernel calls

/**
 * CUDA kernel wrapper function to warm up GPU for accurate benchmarking. Takes no arguments
 */
extern "C" void kernel_WarmUp()
{
    KWarmUp <<<1, 1>>> ();
}

/**
 * CUDA kernel wrapper function for permutation CUDA kernel. Takes input and output image vectors, permutation vectors, thread grid, thread block and mode of operation as arguments
 */
extern "C" void Wrap_RotatePerm(uint8_t* in, uint8_t* out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block, const int mode)
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

/**
 * CUDA kernel wrapper function for diffusion and self - xor CUDA kernels. Takes input and output image vectors, diffusion vectors, image dimensions and all chaotic map parameters  mode of operation as arguments
 */
extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& randRowX, const double*& randRowY, const int dim[], const double r, const int mode, uint32_t diffuse_propagation_factor)
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
        
        DIFF_TD <<<gridRow, block>>> (in, out, randRowX, randRowY, dim[0], r, diffuse_propagation_factor);
        ENC_XOR_LR <<<gridRow, block>>> (out, diffuse_propagation_factor, dim[0]);
       
        
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
        
        DEC_XOR_LR <<<gridRow, block>>> (in, diffuse_propagation_factor, dim[0]);
        DIFF_TD <<<gridRow, block>>> (in, out, randRowX, randRowY, dim[0], r, diffuse_propagation_factor);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
    
        std::printf("\nTime to undiffuse:  %3.6f ms \n", time);
    }
}

/**
 * CUDA kernel wrapper function for caculating sum of image. Takes image vector and resultant sum as arguments 
 */
extern "C" void Wrap_imageSumReduce(uint8_t* __restrict__ image_vec, uint32_t *device_hash_sum, const int dim[])
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
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, image_vec, device_hash_sum, num_items);
  
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  //Run the reduction function to get the sum of the image
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, image_vec, device_hash_sum, num_items);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("\nTime to reduce sum:  %3.6f ms \n", time);
}


