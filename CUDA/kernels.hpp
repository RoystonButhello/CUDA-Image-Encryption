#pragma once
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" void kernel_WarmUp();

extern "C" void Enc_Permute(uint8_t *, uint8_t *, int*, int*, const dim3 &, const dim3 &);

extern "C" void Dec_Permute(uint8_t *, uint8_t *, int*, int*, const dim3 &, const dim3 &);

//auto start = steady_clock::now();
//cout << "Permute: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
