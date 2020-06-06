#ifndef KERNEL_H /*To ensure no errors if the header file is included twice*/
#define KERNEL_H
#include <cuda_runtime_api.h>
#include <cuda.h>

/**
 * This header file contains CUDA kernel wrapper function prototypes that are used to interface between the CUDA kernels and the rest of the program
 */

/**
 * Gets GPU ready to perform computation. Helps achieve accurate GPU benchmarking
 */
extern "C" void run_WarmUp(dim3 blocks,dim3 block_size);

/**
 * Rotates image rows and columns. Based on Arnold Cat Map. Accepts images of dimensions N x N and N x M
 */
extern "C" void run_EncGenCatMap(uint8_t* in,uint8_t* out,const uint32_t* __restrict__ colRotate,const uint32_t* __restrict__ rowRotate,dim3 blocks,dim3 block_size);

/**
 * Unrotates image rows and columns. Based on Arnold Cat Map. Accepts images of dimensions N x N and N x M
 */
extern "C" void run_DecGenCatMap(uint8_t* in,uint8_t* out,const uint32_t* __restrict__ colRotate,const uint32_t* __restrict__ rowRotate,dim3 blocks,dim3 block_size);

/**
 * Swaps image rows and columns. Accepts images of dimensions N x N and N x M
 */
extern "C" void run_encRowColSwap(uint8_t* img_in,uint8_t* img_out,const uint32_t* __restrict__ rowSwapLUT,const uint32_t* __restrict__ colSwapLUT,dim3 blocks,dim3 block_size);

/**
 * Unwaps image rows and columns. Accepts images of dimensions N x N and N x M
 */
extern "C" void run_decRowColSwap(uint8_t* img_in,uint8_t* img_out,const uint32_t* __restrict__ rowSwapLUT,const uint32_t* __restrict__ colSwapLUT,dim3 blocks,dim3 block_size);
 
#endif

