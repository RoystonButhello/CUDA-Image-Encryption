// Wrappers and chaotic maps
#ifndef KERNELS_H
#define KERNELS_H
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cub-1.8.0/cub/cub.cuh"

// CUDA Kernel Wrapper Functions

/**
 * CUDA kernel wrapper function to warm up GPU for accurate benchmarking. Takes no arguments
 */
extern "C" void kernel_WarmUp();

/**
 * CUDA kernel wrapper function for permutation CUDA kernel. Takes input and output image vectors, permutation vectors, thread grid, thread block and mode of operation as arguments
 */
extern "C" void Wrap_RotatePerm(uint8_t* in, uint8_t* out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block, const int mode);

/**
 * CUDA kernel wrapper function for diffusion and self - xor CUDA kernels. Takes input and output image vectors, diffusion vectors, image dimensions and all chaotic map parameters  mode of operation as arguments
 */
extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& randRowX, const double*& randRowY, const int dim[], const double r, const int mode, uint32_t diffuse_propagation_factor); 

/**
 * CUDA kernel wrapper function for caculating sum of image. Takes image vector and resultant sum as arguments 
 */
extern "C" void Wrap_imageSumReduce(uint8_t* __restrict__ image_vec, uint32_t *device_hash_sum, const int dim[]);

#endif


