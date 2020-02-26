#ifndef KERNEL_H /*To ensure no errors if the header file is included twice*/
#define KERNEL_H
#include <cuda_runtime_api.h>
#include <cuda.h>

/*Phase 4 start Arnold mapping*/
extern "C" void run_ArMapImg(uint8_t *in, uint8_t *out,dim3 blocks,dim3 block_size);
extern "C" void run_WarmUp(dim3 blocks,dim3 block_size);

#endif

