#pragma once
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Warm-up Kernel
__global__ void KWarmUp()
{
    return;
}

// Permutation Kernel
__global__ void Enc_GenCatMap(uint8_t*in, uint8_t*out, int*colRotate, int*rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int InDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * 3 + threadIdx.x;
    int OutDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * 3 + threadIdx.x;
    out[OutDex] = in[InDex];
}

// Unpermutation Kernel
__global__ void Dec_GenCatMap(uint8_t*in, uint8_t*out, int*colRotate, int*rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int OutDex = ((gridDim.y) * blockIdx.x + blockIdx.y) * 3 + threadIdx.x;
    int InDex = ((gridDim.y) * ((blockIdx.x + colShift) % gridDim.x) + (blockIdx.y + rowShift) % gridDim.y) * 3 + threadIdx.x;
    out[OutDex] = in[InDex];
}

extern "C" void kernel_WarmUp()
{
    KWarmUp << <1, 1 >> > ();
}

extern "C" void Enc_Permute(uint8_t * in, uint8_t * out, int* colRotate, int* rowRotate, const dim3 &grid, const dim3 &block)
{
    Enc_GenCatMap << <grid, block >> > (in, out, colRotate, rowRotate);
}

extern "C" void Dec_Permute(uint8_t * in, uint8_t * out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block)
{
    Dec_GenCatMap << <grid, block >> > (in, out, colRotate, rowRotate);
}