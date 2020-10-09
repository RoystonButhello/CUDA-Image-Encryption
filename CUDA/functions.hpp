// Ancillary functions

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cstdint>
#include <random>
#include <chrono>
#include "Classes.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace chrono;

/* CUDA Kernel Wrapper Function Declarations */

// GPU Warmup
extern "C" void Wrap_WarmUp();

// Permutation
extern "C" void Wrap_Permutation(uint8_t * in, uint8_t * out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block, const bool mode);

// Diffusion
extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& randRowX, const double*& randRowY, const int dim[], const double r, const bool mode, uint32_t diff_propfac);

// Reduce image matrix to sum of contents
extern "C" void Wrap_ImageReduce(uint8_t * __restrict__ image_vec, uint32_t * device_hash_sum, const int dim[]);


/* Functions that generate a value using the Mersenne Twister PRNG */

static inline int getRandUInt32(int LOWER_BOUND, int UPPER_BOUND)
{
    std::random_device r;
    std::seed_seq seed{ r(), r(), r(), r(), r(), r(), r(), r() };
    std::mt19937 seeder(seed);
    std::uniform_int_distribution<uint32_t> intGen(LOWER_BOUND, UPPER_BOUND);
    return (uint32_t)intGen(seeder);
}

static inline double getRandDouble(double LOWER_BOUND, double UPPER_BOUND)
{
    std::random_device r;
    std::seed_seq seed{ r(), r(), r(), r(), r(), r(), r(), r() };
    std::mt19937 seeder(seed);
    std::uniform_real_distribution<double> realGen(LOWER_BOUND, UPPER_BOUND);
    return (double)realGen(seeder);
}

static inline Chaos getRandCRNG(int LOWER_BOUND, int UPPER_BOUND)
{
    std::random_device r;
    std::seed_seq seed{ r(), r(), r(), r(), r(), r(), r(), r() };
    std::mt19937 seeder(seed);
    std::uniform_int_distribution<int> intGen(LOWER_BOUND, UPPER_BOUND);
    return (Chaos)intGen(seeder);
}


/* CRNG State Update functions */

inline void Step_Arnold(double& x, double& y)
{
    auto xtmp = x + y;
    y = x + 2 * y;
    x = xtmp - (int)xtmp;
    y = y - (int)y;
}

inline void Step_LM(double& x, double& y, const double& r)
{
    x = r * (3 * y + 1) * x * (1 - x);
    y = r * (3 * x + 1) * y * (1 - y);
}

inline void Step_SLMM(double& x, double& y, const double& alpha, const double& beta)
{
    x = alpha * (sin(M_PI * y) + beta) * x * (1 - x);
    y = alpha * (sin(M_PI * x) + beta) * y * (1 - y);
    return;
}

inline void Step_LASM(double& x, double& y, const double& myu)
{
    x = sin(M_PI * myu * (y + 3) * x * (1 - x));
    y = sin(M_PI * myu * (x + 3) * y * (1 - y));
    return;
}

inline void Step_LALM(double& x, double& y, const double& myu)
{
    auto xtmp = myu * (y * 3) * x * (1 - x);
    x = 4 * xtmp * (1 - xtmp);
    auto ytmp = myu * (x + 3) * y * (1 - y);
    y = 4 * ytmp * (1 - ytmp);
    return;
}

inline void CRNGUpdate(double& x, double& y, const double& alpha, const double& beta, const double& myu, const double& r, Chaos Map)
{
    switch (Map)
    {
    case Chaos::Arnold: Step_Arnold(x, y);
        break;

    case Chaos::LM: Step_LM(x, y, r);
        break;

    case Chaos::SLMM: Step_SLMM(x, y, alpha, beta);
        break;

    case Chaos::LASM: Step_LASM(x, y, myu);
        break;

    case Chaos::LALM: Step_LALM(x, y, myu);
        break;

    default: std::cout << "\nInvalid CRNG Option!\nExiting...";
        std::exit(0);
    }
}


/* Other functions */

inline size_t CRNGVecSize(std::vector<CRNG>& vec)
{
    size_t size = 0;
    auto unit = sizeof(double);
    for (CRNG temp : vec)
    {
        switch (temp.map)
        {
        case Chaos::Arnold: size += (unit * 2); break;
        case Chaos::SLMM: size += (unit * 4); break;
        default: size += (unit * 3); break;
        }
    }
    return size;
}

inline void timeSince(time_point<steady_clock> start, string text)
{
    auto value = (int)duration_cast<microseconds>(steady_clock::now() - start).count();
    if (value > 1000)
    {
        printf("\n%s: %3.6fms", text.c_str(), value / 1000.0);
    }
    else
    {
        printf("\n%s: %dus", text.c_str(), value);
    }
}

#endif
