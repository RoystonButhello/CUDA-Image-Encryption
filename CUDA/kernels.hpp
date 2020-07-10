// Wrappers and chaotic maps
#ifndef KERNELS_H
#define KERNELS_H
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Kernel Wrapper Functions
extern "C" void kernel_WarmUp();
extern "C" void Wrap_RotatePerm(uint8_t*, uint8_t*, int*, int*, const dim3&, const dim3&, const int mode);
extern "C" void Wrap_Diffusion(uint8_t*&, uint8_t*&, uint32_t host_sum_plain, const double*& , const double*&, const int [], const double, const int mode);
extern "C" void Wrap_imageSum(uint8_t *&image_vec, uint32_t *sum, const int dim[]);
#endif

