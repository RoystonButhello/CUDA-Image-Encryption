//CUDA kernels and launching functions

#include <cstdint>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cub/cub.cuh"

using namespace std;
using namespace cub;
using namespace thrust;

// Warm-up CUDA kernel for accurate benchmarking
__global__ void WarmUp()
{
    return;
}

// Permutation by Rotation (ENC)
__global__ void ENC_PERM(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int InDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * blockDim.x + threadIdx.x;
    int OutDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * blockDim.x + threadIdx.x;
    out[OutDex] = in[InDex];
}

// Permutation by Rotation (DEC)
__global__ void DEC_PERM(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int InDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * blockDim.x + threadIdx.x;
    int OutDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * blockDim.x + threadIdx.x;
    out[InDex] = in[OutDex];
}

// Row-wise Gray Level Transform using 2D Logistic Map
__global__ void DIFF_TD(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xRow, const double* __restrict__ yRow, const int rows, const double r, const uint32_t propfac)
{
    // Initialize parameters
    double x = xRow[blockIdx.x];
    double y = yRow[blockIdx.x];
    const int stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread diffuses one channel of a column
    for (int i = 0; i < rows; i++, idx += stride)
    {
        x = r * (3 * y + 1) * x * (1 - x);
        y = r * (3 * x + 1) * y * (1 - y);
        out[idx] = in[idx] ^ propfac ^ (uint8_t)(x * 256);
    }
}

// Independent diffusion of each row (ENC)
__global__ void ENC_DIFF_LR(uint8_t* __restrict__ in, const int cols, const uint32_t propfac)
{
    // Initialize parameters
    int prev = cols * blockIdx.x * blockDim.x + threadIdx.x;
    int curr = prev + blockDim.x;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        in[curr] = in[curr] ^ propfac ^ in[prev];
        prev = curr;
        curr += blockDim.x;
    }
}

// Independent diffusion of each row (DEC)
__global__ void DEC_DIFF_LR(uint8_t* __restrict__ img, const int cols, uint32_t propfac)
{
    // Initialize parameters
    int curr = cols * blockIdx.x * blockDim.x + threadIdx.x + (cols - 1) * blockDim.x;
    int next = curr - blockDim.x;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        img[curr] = img[curr] ^ propfac ^ img[next];
        curr = next;
        next -= blockDim.x;
    }
}

/* Wrappers for kernel launches */

// GPU Warmup
extern "C" void Wrap_WarmUp()
{
    WarmUp <<<1, 1 >>> ();
}

// Permutation
extern "C" void Wrap_Permutation(uint8_t * in, uint8_t * out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block, const bool mode)
{
    if (mode == true)
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        ENC_PERM << <grid, block >> > (in, out, colRotate, rowRotate);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::printf("\nPermute Kernel (GPU):  %3.6f ms \n", time);
    }

    else
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        DEC_PERM << <grid, block >> > (in, out, colRotate, rowRotate);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::printf("\nPermute Kernel (GPU):  %3.6f ms \n", time);
    }
}

// Diffusion
extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& xRow, const double*& yRow, const int dim[], const double r, const bool mode, uint32_t propfac)
{
    // Set grid and block size
    const dim3 gridCol(dim[0], 1, 1);
    const dim3 gridRow(dim[1], 1, 1);
    const dim3 block(dim[2], 1, 1);

    if (mode == true)
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        DIFF_TD << <gridRow, block >> > (in, out, xRow, yRow, dim[0], r, propfac);
        ENC_DIFF_LR << <gridRow, block >> > (out, dim[0], propfac);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::printf("\nDiffuse kernels (GPU):  %3.6f ms \n", time);
    }

    else
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        DEC_DIFF_LR << <gridRow, block >> > (in, dim[0], propfac);
        DIFF_TD << <gridRow, block >> > (in, out, xRow, yRow, dim[0], r, propfac);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::printf("\nDiffuse kernels (GPU):  %3.6f ms \n", time);
    }
}

// Reduce image matrix to sum of contents
extern "C" void Wrap_ImageReduce(uint8_t * __restrict__ img, uint32_t * sum, const int dim[])
{
    int num_items = dim[0] * dim[1] * dim[2];
    void* temp_ptr = NULL;
    size_t temp_mem_size = 0;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Check how much temporary storage is needed
    cub::DeviceReduce::Sum(temp_ptr, temp_mem_size, img, sum, num_items);

    // Allocate VRAM
    cudaMalloc(&temp_ptr, temp_mem_size);

    // Run the reduction function
    cub::DeviceReduce::Sum(temp_ptr, temp_mem_size, img, sum, num_items);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\nImage Reduction (GPU):  %3.6f ms \n", time);
}


