// CUDA-related function definitions
#pragma once
#include <stdint.h>
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
__global__ void DIFF_TD(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xRow, const double* __restrict__ yRow, const int rows, const double r)
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
        out[idx] = in[idx] ^ (uint8_t)(x * 256);
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
         in[curr] ^= in[prev];
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
        img[curr] ^= img[next];
        curr = next;
        next -= blockDim.x;
    }
}

// Wrappers for kernel calls
extern "C" void kernel_WarmUp()
{
    KWarmUp << <1, 1 >> > ();
}

extern "C" void Wrap_RotatePerm(uint8_t * in, uint8_t * out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block, const int mode)
{
    if (mode == 0)
    {
        ENC_RotatePerm << <grid, block >> > (in, out, colRotate, rowRotate);
        ENC_RotatePerm << <grid, block >> > (out, in, colRotate, rowRotate);
    }
    else
    {
        DEC_RotatePerm << <grid, block >> > (in, out, colRotate, rowRotate);
        DEC_RotatePerm << <grid, block >> > (out, in, colRotate, rowRotate);
    }
}

extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& randRowX, const double*& randRowY, const int dim[], const double r, const int mode)
{
    // Set grid and block size
    const dim3 gridCol(dim[0], 1, 1);
    const dim3 gridRow(dim[1],1, 1);
    const dim3 block(dim[2], 1, 1);

    if (mode == 0)
    {
        DIFF_TD << <gridRow, block >> > (in, out, randRowX, randRowY, dim[0], r);
        ENC_XOR_LR << <gridRow, block >> > (out, dim[0]);
    }
    else
    {
        DEC_XOR_LR << <gridRow, block >> > (in, dim[0]);
        DIFF_TD << <gridRow, block >> > (in, out, randRowX, randRowY, dim[0], r);
    }
}
