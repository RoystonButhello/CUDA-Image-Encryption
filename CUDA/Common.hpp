#pragma once
#include <iostream>
#include <cstdio>
#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace std;
using namespace thrust;
using namespace cv::cuda;

// Structure for consolidating file-related strings
struct paths
{
    string src = "images/";
    string temp = "temp/";
    string target = "cat.png";
    string fn_img_in = src + target;
    string fn_img_out = fn_img_in + "_ENC";
    string fn_vars = temp + "vars.txt";
}path;

// Generate vector of random real numbers in (0,1): randomReal for diffusion
// Return vector of N random integers in [0,M] for permutation
host_vector<int> genRelocVec(const int M, const int N, vector<double>& randomReal)
{
    //Initiliaze Generators
    double unzero = 0.0000000001;
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

// Permutation Kernel
__global__ void Enc_GenCatMap(const PtrStepSz<uint8_t> in, PtrStepSz<uint8_t> out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int x = (blockIdx.x + colShift) % gridDim.x;
    int y = (blockIdx.y + rowShift) % gridDim.y;
    out.ptr(x)[y] = in.ptr(blockIdx.x)[blockIdx.y];
}

// Unpermutation Kernel
__global__ void Dec_GenCatMap(const PtrStepSz<uint8_t> in, PtrStepSz<uint8_t> out, const int* __restrict__ colRotate, const int* __restrict__ rowRotate)
{
    int colShift = colRotate[blockIdx.y];
    int rowShift = rowRotate[(blockIdx.x + colShift) % gridDim.x];
    int x = (blockIdx.x + colShift) % gridDim.x;
    int y = (blockIdx.y + rowShift) % gridDim.y;
    out.ptr(blockIdx.x)[blockIdx.y] = in.ptr(x)[y];
}

cudaError_t CudaPermute(Mat3b img, const host_vector<int> &U, const host_vector<int> &V)
{
    // Upload rotation vectors to device
    const int M = img.rows, N = img.cols;
    device_vector<int> D_U = U, D_V = V;
    int* ptrU = thrust::raw_pointer_cast(&D_U[0]);
    int* ptrV = thrust::raw_pointer_cast(&D_V[0]);

    // Upload img to device
    GpuMat gpuImgIn(M, N, CV_8UC3);
    GpuMat gpuImgOut(M, N, CV_8UC3);

    // Set grid and block size
    dim3 grid = (M, N, 1);
    dim3 block = (3, 1, 1);

    Enc_GenCatMap << <grid, block >> > (gpuImgIn, gpuImgOut, ptrU, ptrV);

    gpuImgOut.download(img);
    cudaFree(ptrU);
    cudaFree(ptrU);
    return cudaSuccess;
}