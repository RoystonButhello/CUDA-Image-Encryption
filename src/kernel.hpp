#ifndef KERNEL_H /*To ensure no errors if the header file is included twice*/
#define KERNEL_H
#include <cuda_runtime_api.h>
#include <cuda.h>
extern "C" void run_GenCatMap(uint8_t *in, uint8_t *out, uint32_t *colRotate, uint32_t *rowRotate,dim3 blocks,dim3 block_size);
#endif

