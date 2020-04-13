#ifndef KERNEL_H /*To ensure no errors if the header file is included twice*/
#define KERNEL_H
#include <cuda_runtime_api.h>
#include <cuda.h>

/*Phase 4 start Arnold mapping*/
extern "C" void run_ArMapImg(uint8_t *in, uint8_t *out,dim3 blocks,dim3 block_size);
extern "C" void run_WarmUp(dim3 blocks,dim3 block_size);
/*Phase 6 start fractal XORing*/
extern "C" void run_FracXor(uint8_t *in,uint8_t *out,uint8_t *fractal,dim3 blocks,dim3 block_size);
/*Phase 7 start Arnold Map Encryption*/
extern "C" void run_EncGenCatMap(uint8_t *in,uint8_t *out,uint16_t *colRotate,uint16_t *rowRotate,dim3 blocks,dim3 block_size);
/*Phase 8 start Arnold Map Decryption*/
extern "C" void run_DecGenCatMap(uint8_t *in,uint8_t *out,uint16_t *colRotate,uint16_t *rowRotate,dim3 blocks,dim3 block_size);
/*Phase 9 start mapping image to Arnold Map Table*/
extern "C" void run_ArMapTable(uint32_t *in,uint32_t *out,dim3 blocks, dim3 block_size);
extern "C" void run_ArMapTabletoImg(uint8_t *in,uint8_t *out,uint32_t *table,dim3 blocks,dim3 block_size);
/*Phase 10 Miscellaneous*/
extern "C" void run_generateU(double *P,uint16_t *U,double n,dim3 blocks,dim3 block_size);
/*Phase 11 gray level transform*/
extern "C" void run_grayLevelTransform(uint8_t *img_vec, uint16_t *random_array, dim3 blocks, dim3 block_size);
/*Phase 12 row column swapping*/
extern "C" void run_encRowColSwap(uint8_t *img_in,uint8_t *img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,dim3 blocks,dim3 block_size);
extern "C" void run_decRowColSwap(uint8_t *img_in,uint8_t *img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,dim3 blocks,dim3 block_size);

#endif

