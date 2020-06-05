// Wrappers and chaotic maps
#pragma once
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUDA Kernel Wrapper Functions
extern "C" void kernel_WarmUp();
extern "C" void Wrap_RotatePerm(uint8_t*, uint8_t*, int*, int*, const dim3&, const dim3&, const int mode);
extern "C" void Wrap_Diffusion(uint8_t*&, uint8_t*&, const double*& , const double*&, const int [], const double, const int mode);

inline void ArnoldIteration(double& x, double& y)
{
    auto xtmp = x + y;
    y = x + 2 * y;
    x = xtmp - (int)xtmp;
    y = y - (int)y;
    return;
}

inline void Logistic2Dv1Iteration(double& x, double& y, const double &r)
{
    x = r * (3 * y + 1) * x * (1 - x);
    y = r * (3 * x + 1) * y * (1 - y);
    return;
}

inline void Logistic2Dv2Iteration(double& x, double& y, const std::vector<double> &v)
{
    auto xtmp = x;
    x = x * v[0] * (1 - x) + v[2] * y * y;
    y = y * v[1] * (1 - y) + v[3] * xtmp * (xtmp + y);
    return;
}
