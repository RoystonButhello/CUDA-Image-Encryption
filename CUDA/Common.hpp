#pragma once
#include <iostream>
#include <cstdio>
#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "kernels.hpp"

using namespace std;
using namespace chrono;
using namespace thrust;

// Structure for consolidating file-related strings
struct paths
{
    string src = "images/";
    string temp = "temp/";
    string target = "mountain1080.png";
    string fn_img_in = src + target;
    string fn_img_out = fn_img_in + "_ENC";
    string fn_vars = temp + "vars.txt";
}path;

struct params
{
    int rotate_rounds = 8;
}p;

// Generate vector of random real numbers in (0,1): randomReal for diffusion
// Return vector of N random integers in [0,M] for permutation
host_vector<int> genRelocVec(const int M, const int N, vector<double>& randomReal)
{
    //Initiliaze Generators
    double unzero = 0.00000001;
    random_device randev;
    mt19937 seeder(randev());
    uniform_int_distribution<int> intGen(1, 32);
    uniform_real_distribution<double> realGen(unzero, 1);

    //Initiliaze parameters
    auto a = intGen(seeder);
    auto b = intGen(seeder);
    auto c = a * b + 1;
    auto x = realGen(seeder);
    auto y = realGen(seeder);
    auto offset = intGen(seeder);

    //Skip first few values in sequence
    for (int i = 0; i < offset; i++)
    {
        x = fmod(x + a * y, 1) + unzero;
        y = fmod(b * x + c * y, 1) + unzero;
    }

    //Generate vector of real numbers in the interval (0,1)
    for (int i = 0, limit = M * N; i < limit; i++)
    {
        x = fmod(x + a * y, 1) + unzero;
        y = fmod(b * x + c * y, 1) + unzero;
        randomReal[i * 2] = x;
        randomReal[i * 2 + 1] = y;
    }

    host_vector<int> relocVec(N);
    uniform_int_distribution<int> offsetGen(1, N * (M - 1));
    auto vec_offset = offsetGen(seeder);
    int exp = (int)pow(10, 8);
    for (int i = 0; i < N; i++)
    {
        relocVec[i] = (int)(randomReal[vec_offset + i] * exp) % M;
    }
    return relocVec;
}

cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], const host_vector<int>& U, const host_vector<int>& V)
{
    // Upload rotation vectors to device
    device_vector<int> D_U = U, D_V = V;
    int* ptrU = thrust::raw_pointer_cast(&D_U[0]);
    int* ptrV = thrust::raw_pointer_cast(&D_V[0]);
    
    // Set grid and block data_size
    const dim3 grid(dim[0], dim[1], 1);
    const dim3 block(dim[2], 1, 1);

    auto start = steady_clock::now();
    // Call Permutation Kernel
    for (int i = 0; i < p.rotate_rounds; i++)
    {
        Enc_Permute(d_img, d_imgtmp, ptrU, ptrV, grid, block);
         swap(d_img, d_imgtmp);
    }
    cout << "Permute: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
    return cudaSuccess;
}