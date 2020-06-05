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

// ENC::SELF-XOR (top-down)
__global__ void ENC_XOR_TD(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xRow, const int rows)
{
    // Initialize parameters
    const int stride = gridDim.x * blockDim.x;
    int prev = blockIdx.x * blockDim.x + threadIdx.x;
    int curr = prev + stride;

    // Initialize top row
    out[prev] = in[prev] ^ (uint8_t)(xRow[blockIdx.x] * 256);

    // Each thread diffuses one channel of a column
    for (int i = 1; i < rows; i++)
    {
        out[curr] = in[curr] ^ out[prev];
        prev = curr;
        curr += stride;
    }
}

// ENC::SELF-XOR (left-right)
__global__ void ENC_XOR_LR(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xCol, const int cols)
{
    // Initialize parameters
    double x = xCol[blockIdx.x];
    const int stride = blockDim.x;
    int prev = cols * blockIdx.x * blockDim.x + threadIdx.x;
    int curr = prev + stride;

    // Initialize leftmost column
    out[prev] = in[prev] ^ (uint8_t)(xCol[blockIdx.x] * 256);

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        out[curr] = in[curr] ^ out[prev];
        prev = curr;
        curr += stride;
    }
}

// ENC::SELF-XOR (bottom-up)
__global__ void ENC_XOR_BU(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xRow, const int rows)
{
    // Initialize parameters
    const int stride = gridDim.x * blockDim.x;
    int prev = blockIdx.x * blockDim.x + threadIdx.x + (rows - 1) * stride;
    int curr = prev - stride;

    // Initialize bottom row
    out[prev] = in[prev] ^ (uint8_t)(xRow[blockIdx.x] * 256);

    // Each thread diffuses one channel of a column
    for (int i = 1; i < rows; i++)
    {
        out[curr] = in[curr] ^ out[prev];
        prev = curr;
        curr -= stride;
    }
}

// ENC::SELF-XOR (right-left)
__global__ void ENC_XOR_RL(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, const double* __restrict__ xCol, const int cols)
{
    // Initialize parameters
    double x = xCol[blockIdx.x];
    const int stride = blockDim.x;
    int prev = cols * blockIdx.x * blockDim.x + threadIdx.x + (cols - 1) * stride;
    int curr = prev - stride;

    // Initialize rightmost column
    out[prev] = in[prev] ^ (uint8_t)(xCol[blockIdx.x] * 256);

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        out[curr] = in[curr] ^ out[prev];
        prev = curr;
        curr -= stride;
    }
}

// DEC::SELF-XOR (top-down)
__global__ void DEC_XOR_TD(uint8_t* __restrict__ img, const double* __restrict__ xRow, const int rows)
{
    // Initialize parameters
    const int stride = gridDim.x * blockDim.x;
    int curr = blockIdx.x * blockDim.x + threadIdx.x + (rows - 1) * stride;
    int next = curr - stride;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < rows; i++)
    {
        img[curr] ^= img[next];
        curr = next;
        next -= stride;
    }

    // Finalize bottom row
    img[curr] ^= (uint8_t)(xRow[blockIdx.x] * 256);
}

// DEC::SELF-XOR (left-right)
__global__ void DEC_XOR_LR(uint8_t* __restrict__ img, const double* __restrict__ xCol, const int cols)
{
    // Initialize parameters
    const int stride = blockDim.x;
    int curr = cols * blockIdx.x * blockDim.x + threadIdx.x + (cols - 1) * stride;
    int next = curr - stride;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        img[curr] ^= img[next];
        curr = next;
        next -= stride;
    }

    // Finalize leftmost column
    img[curr] ^= (uint8_t)(xCol[blockIdx.x] * 256);
}

// DEC::SELF-XOR (bottom-up)
__global__ void DEC_XOR_BU(uint8_t* __restrict__ img, const double* __restrict__ xRow, const int rows)
{
    // Initialize parameters
    const int stride = gridDim.x * blockDim.x;
    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    int next = curr + stride;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < rows; i++)
    {
        img[curr] ^= img[next];
        curr = next;
        next += stride;
    }

    // Finalize bottom row
    img[curr] ^= (uint8_t)(xRow[blockIdx.x] * 256);
}

// DEC::SELF-XOR (right-left)
__global__ void DEC_XOR_RL(uint8_t* __restrict__ img, const double* __restrict__ xCol, const int cols)
{
    // Initialize parameters
    const int stride = blockDim.x;
    int curr = cols * blockIdx.x * blockDim.x + threadIdx.x;
    int next = curr + stride;

    // Each thread diffuses one channel of a row
    for (int i = 1; i < cols; i++)
    {
        img[curr] ^= img[next];
        curr = next;
        next += stride;
    }

    // Finalize rightmost column
    img[curr] ^= (uint8_t)(xCol[blockIdx.x] * 256);
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

extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& randRowX, const double*& randColX, const int dim[], const double r, const int mode)
{
    // Set grid and block size
    const dim3 gridCol(dim[0], 1, 1);
    const dim3 gridRow(dim[1],1, 1);
    const dim3 block(dim[2], 1, 1);

    if (mode == 0)
    {
        ENC_XOR_TD << <gridRow, block >> > (in, out, randRowX, dim[0]);
        ENC_XOR_LR << <gridCol, block >> > (out, in, randColX, dim[1]);
        ENC_XOR_BU << <gridRow, block >> > (in, out, randRowX, dim[0]);
        ENC_XOR_RL << <gridCol, block >> > (out, in, randColX, dim[1]);
    }
    else
    {
        DEC_XOR_RL << <gridCol, block >> > (in, randColX, dim[1]);
        DEC_XOR_BU << <gridRow, block >> > (in, randRowX, dim[0]);
        DEC_XOR_LR << <gridCol, block >> > (in, randColX, dim[1]);
        DEC_XOR_TD << <gridRow, block >> > (in, randRowX, dim[0]);
    }
}
