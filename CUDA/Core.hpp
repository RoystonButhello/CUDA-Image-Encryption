// Top-level encryption functions
#pragma once
#include <random>
#include <chrono>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "functions.hpp"
#include "structures.hpp"

using namespace cv;
using namespace chrono;
using namespace thrust;

// Store generated parameters in vectors as File I/O is somehow proving to be the most trying part of this code
vector<Permuter> pVec;
vector<Diffuser> dVec;

// Initiliaze parameters
void Initialize()
{
    string file;
    cout << "Filename: ";
    cin >> file;
    cout << "Rounds: ";
    scanf("%d", &cfg.rounds);
    cfg.rounds = (cfg.rounds > 0) ? cfg.rounds : 2;
    cfg.rotations = 2;
    buildPaths(file);
}

// Generate vector of N random numbers in [0,M]
int* getPermVec(const int M, const int N, Permuter &crng, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENC)
    {
        crng.map = Chaos::Logistic2Dv1;
        random_device randev;
        uniform_real_distribution<double> realGen(0, 1);
        crng.core = randev();
        crng.x = realGen(randev);
        crng.y = realGen(randev);
    }

    //Initiliaze Parameters
    mt19937 seeder(crng.core);
    double x = crng.x;
    double y = crng.y;

    host_vector<int> ranVec(N);
    const int exp = (int)pow(10, 9);
    int i = 0;

    auto start = steady_clock::now();
    switch (crng.map)
    {
        case Chaos::Arnold:
        {
            for (i = 0; i < 32; i++)
            { ArnoldIteration(x, y);}
            for (i = 0; i < N; i++)
            {
                ArnoldIteration(x, y);
                ranVec[i] = (int)(x * exp) % M;
            }
            break;
        }
        case Chaos::Logistic2Dv1:
        {
            uniform_real_distribution<double> rGen(1.18, 1.19);
            const double r = rGen(seeder);

            for (i = 0; i < 32; i++)
            { Logistic2Dv1Iteration(x, y, r);}
            for (i = 0; i < N; i++)
            {
                Logistic2Dv1Iteration(x, y, r);
                ranVec[i] = (int)(x * exp) % M;
            }
            break;
        }
        case Chaos::Logistic2Dv2:
        {
            uniform_real_distribution<double> u1Gen(2.75, 3.4), u2Gen(2.75, 3.45), v1Gen(0.15, 0.21), v2Gen(0.13, 0.15);
            const vector<double> args{ u1Gen(seeder), u2Gen(seeder), v1Gen(seeder),v2Gen(seeder) };

            for (i = 0; i < 32; i++)
            { Logistic2Dv2Iteration(x, y, args);
            }
            for (i = 0; i < N; i++)
            {
                Logistic2Dv2Iteration(x, y, args);
                ranVec[i] = (int)(x * exp) % M;\
            }
            break;
        }
    }
    cout << "PERM. CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";

    device_vector<int> dVec = ranVec;
    return (int *)(thrust::raw_pointer_cast(&dVec[0]));
}

// Generate 2 vectors of N random numbers in (0,1] each
void getDiffVecs(host_vector<double> &xVec, host_vector<double> &yVec, const int M, const int N, Diffuser& crng, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENC)
    {
        crng.map = Chaos::Logistic2Dv1;
        random_device randev;
        uniform_real_distribution<double> realGen(0, 1);
        crng.core = randev();
        crng.x = realGen(randev);
        crng.y = realGen(randev);
    }

    //Initiliaze Parameters
    mt19937 seeder(crng.core);
    double x = crng.x;
    double y = crng.y;

    const int exp = (int)pow(10, 9);
    int i = 0;
    auto start = steady_clock::now();
    switch (crng.map)
    {
        case Chaos::Arnold:
        {
            for (i = 0; i < 32; i++)
            {
                ArnoldIteration(x, y);
            }
            for (i = 0; i < N; i++)
            {
                ArnoldIteration(x, y);
                xVec[i] = x;
                yVec[i] = y;
            }
            break;
        }
        case Chaos::Logistic2Dv1:
        {
            uniform_real_distribution<double> rGen(1.18, 1.19);
            const double r = rGen(seeder);

            for (i = 0; i < 32; i++)
            {
                Logistic2Dv1Iteration(x, y, r);
            }
            for (i = 0; i < N; i++)
            {
                Logistic2Dv1Iteration(x, y, r);
                xVec[i] = x;
                yVec[i] = y;
            }
            break;
        }
        case Chaos::Logistic2Dv2:
        {
            uniform_real_distribution<double> u1Gen(2.75, 3.4), u2Gen(2.75, 3.45), v1Gen(0.15, 0.21), v2Gen(0.13, 0.15);
            const vector<double> args{ u1Gen(seeder), u2Gen(seeder), v1Gen(seeder),v2Gen(seeder) };

            for (i = 0; i < 32; i++)
            {
                Logistic2Dv2Iteration(x, y, args);
            }
            for (i = 0; i < N; i++)
            {
                Logistic2Dv2Iteration(x, y, args);
                xVec[i] = x;
                yVec[i] = y;
            }
            break;
        }
    }
    cout << "DIFF. CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
}

cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m)
{
    // Generate permutation vectors
    auto ptrU = getPermVec(dim[0], dim[1], perm[0], m);
    auto ptrV = getPermVec(dim[1], dim[0], perm[1], m);
    
    // Set grid and block data_size
    const dim3 grid(dim[0], dim[1], 1);
    const dim3 block(dim[2], 1, 1);

    auto start = steady_clock::now();
    Wrap_RotatePerm(d_img, d_imgtmp, ptrU, ptrV, grid, block,int(m));
    cout << "Permutation: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";

    return cudaDeviceSynchronize();
}

cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m)
{
    // Initiliaze diffusion vectors
    host_vector<double> randRowX(dim[1]), randRowY(dim[1]);

    getDiffVecs(randRowX, randRowY, dim[0], dim[1], diff, m);

    device_vector<double> DRowX = randRowX, DRowY = randRowY;

    const double* rowXptr = (double*)(thrust::raw_pointer_cast(&DRowX[0]));
    const double* rowYptr = (double*)(thrust::raw_pointer_cast(&DRowY[0]));

    // Initiliaze control parameter
    if (m == Mode::ENC)
    {
        random_device rd;
        uniform_real_distribution<double> rGen(1.18, 1.19);
        diff.r = rGen(rd);
    }

    auto start = steady_clock::now();
    Wrap_Diffusion(d_img, d_imgtmp, rowXptr, rowYptr, dim, diff.r, int(m));
    cout << "Diffusion: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";
    swap(d_img, d_imgtmp);
   
    return cudaDeviceSynchronize();
}

int Encrypt()
{
    kernel_WarmUp();
    Initialize();

    // Read the file and confirm it's been opened
    Mat img = imread(path.fn_img, 1);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };

    // Upload image and LUTs to device
    uint8_t* d_img, * d_imgtmp;
    int *gpu_u; 
    int *gpu_v;
    
    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
      
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);
    
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);
    
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);
    
    // Show original image
    //imshow("Original", img);

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------ENCRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n\n";

    cudaError_t cudaStatus;

    // Encryption rounds
    for (int i = 0; i < cfg.rounds; i++)
    {
        cout << "X------ROUND " << i + 1 << "------X\n";

        // Permute Image
        for (int j = 0; j < cfg.rotations; j++)
        {
            cout << "\n     --Rotation " << j + 1 << "--     \n";
            cudaStatus = CudaPermute(d_img, d_imgtmp, dim, Mode::ENC);
            if (cudaStatus != cudaSuccess)
            {
                cout<<"\ncudaStatus = "<<cudaStatus;
                cerr << "ENC_Permutation Failed!\n";
                return -1;
            }
            pVec.push_back(perm[0]);
            pVec.push_back(perm[1]);
        }

        //Diffuse image
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, Mode::ENC);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "ENC_Diffusion Failed!\n";
            return -1;
        }
        dVec.push_back(diff);
    }

    // Display encrypted image
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    imwrite(path.fn_img_enc, img);
    //imshow("Encrypted", img);

    cudaDeviceReset();
    return 0;
}

int Decrypt()
{
    kernel_WarmUp();

    // Read the file and confirm it's been opened
    Mat img = imread(path.fn_img_enc, -1);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };

    // Upload image and LUTs to device
    uint8_t* d_img, * d_imgtmp;
    int *gpu_u; 
    int *gpu_v;
    
    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
      
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);
    
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);
    
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);
    

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------DECRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n\n";
    
    cudaError_t cudaStatus;

    // Decryption rounds
    for (int i = cfg.rounds - 1; i >= 0; i--)
    {
        cout << "X------ROUND " << i + 1 << "------X\n";

        //Undiffuse image
        diff = dVec[i];
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, Mode::DEC);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "DEC_Diffusion Failed!\n";
            return -1;
        }
        
        //Unpermute image
        for (int j = cfg.rotations - 1, idx = 4 * i + 2 * j; j >= 0; j--, idx-=2)
        {
            cout << "\n     --Rotation " << j + 1 << "--     \n";
            perm[0] = pVec[idx];
            perm[1] = pVec[idx+1];
            cudaStatus = CudaPermute(d_img, d_imgtmp, dim, Mode::DEC);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "DEC_Permutation Failed!\n";
                return -1;
            }
        }
    }

    // Display decrypted image
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    imwrite(path.fn_img_dec, img);
    //imshow("Decrypted", img);

    cudaDeviceReset();
    waitKey(0);
    return 0;
}

