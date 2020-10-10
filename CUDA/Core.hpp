// Top-level encryption functions

#ifndef CORE_H
#define CORE_H


#define DEBUG_KEY                        1
#define PRINT_IMAGES                     1

#define XY_LOWER_LIMIT                   0.1
#define XY_UPPER_LIMIT                   0.7
#define ALPHA_LOWER_LIMIT                0.905
#define ALPHA_UPPER_LIMIT                0.985
#define BETA_LOWER_LIMIT                 2.97
#define BETA_UPPER_LIMIT                 3.00
#define MYU_LOWER_LIMIT                  0.50
#define MYU_UPPER_LIMIT                  0.90
#define R_LOWER_LIMIT                    1.15
#define R_UPPER_LIMIT                    1.17
#define MAP_LOWER_LIMIT                  0
#define MAP_UPPER_LIMIT                  4
#define PERMUTE_PROPAGATION_LOWER_LIMIT  4000
#define PERMUTE_PROPAGATION_UPPER_LIMIT  5000
#define DIFFUSE_PROPAGATION_LOWER_LIMIT  8000
#define DIFFUSE_PROPAGATION_UPPER_LIMIT  10000  
#define OFFSET_LOWER_LIMIT              0.00001  
#define OFFSET_UPPER_LIMIT              0.00002

#include <cstdio>   //printf
#include <openssl/sha.h> //sha256
#include <opencv2/opencv.hpp> //Convenient Image I/O
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thrust/host_vector.h> // Convenient vector transfer between host and device
#include <thrust/device_vector.h>
#include "functions.hpp"

using namespace cv;
using namespace std;
using namespace chrono;
using namespace thrust;

// Store generated parameters in vectors instead of writing to disk as File I/O is somehow proving to be the most trying part of this code
std::vector<CRNG> pVec, dVec;

// Initiliaze parameters
void Initialize()
{
    string file;
    cout << "Filename: ";
    cin >> file;
    path.buildPaths(file);

    cout << "Rounds: ";
    scanf("%d", &config.rounds);
    config.rounds = (config.rounds > 0) ? config.rounds : 2;
    config.rotations = 2;

    cout << endl;
}

// Returns number of digits in provided value
int getNumOfDigits(uint32_t value)
{
    int power = 0;
    while (value > 0)
    {
        value = value / 10;
        ++power;
    }
    return power;
}

// Generates offset (<0.1) using the given integer
double getParameterOffset(uint32_t value)
{
    double divisor = pow(10, ((double)getNumOfDigits(value) + 2));
    return (value > 0) ? (double)value / divisor : -0.05;
}

// Modifies permutation parameters for a given permutation round
void modifyParameters(CRNG& crng, double modifier, double offset)
{
    switch (crng.map)
    {
    case Chaos::Arnold:
    {
        crng.x += modifier + offset;
        crng.y += modifier + offset;
    }
    break;

    case Chaos::LM:
    {
        crng.x += modifier + offset;
        crng.y += modifier + offset;
        crng.r += modifier + offset;
    }
    break;

    case Chaos::SLMM:
    {
        crng.x += modifier + offset;
        crng.y += modifier + offset;
        crng.alpha += modifier + offset;
        crng.beta += modifier + offset;
    }
    break;

    case Chaos::LASM:
    {
        crng.x += modifier + offset;
        crng.y += modifier + offset;
        crng.myu += modifier + offset;
    }
    break;

    case Chaos::LALM:
    {
        crng.x += modifier + offset;
        crng.y += modifier + offset;
        crng.myu += modifier + offset;
    }
    break;
    }
}

// Generates vector of N random numbers in [0,M]
int* getPermVec(const int M, const int N, CRNG& permute, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENC)
    {
        permute.map = getRandCRNG(MAP_LOWER_LIMIT, MAP_UPPER_LIMIT);
        permute.x = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        permute.y = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        if (permute.map == Chaos::LM)
        {
            permute.r = getRandDouble(R_LOWER_LIMIT, R_UPPER_LIMIT);
        }
        else if (permute.map == Chaos::SLMM)
        {
            permute.alpha = getRandDouble(ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT);
            permute.beta = getRandDouble(BETA_LOWER_LIMIT, BETA_UPPER_LIMIT);
        }
        else if (permute.map != Chaos::Arnold)
        {
            permute.myu = getRandDouble(MYU_LOWER_LIMIT, MYU_UPPER_LIMIT);
        }
        offset.perm_offset = getRandDouble(OFFSET_LOWER_LIMIT, OFFSET_UPPER_LIMIT);
        modifyParameters(permute, offset.perm_modifier, offset.perm_offset);
    }

    //Initialize Parameters
    Chaos map = permute.map;
    double x = permute.x;
    double y = permute.y;
    const double alpha = permute.alpha;
    const double beta = permute.beta;
    const double myu = permute.myu;
    const double r = permute.r;

    host_vector<int> ranVec(N);
    const int exp = (int)pow(10, 9);

    auto start = steady_clock::now();

    for (int i = 0; i < N; ++i)
    {
        CRNGUpdate(x, y, alpha, beta, myu, r, map);
        ranVec[i] = (int)(x * exp) % M;
    }

    timeSince(start, "Permutation CRNG");

    device_vector<int> dVec = ranVec;
    return (int*)(thrust::raw_pointer_cast(&dVec[0]));
}

// Generates vector of M random numbers in (0.0,1.0)
void getDiffVecs(host_vector<double>& xVec, host_vector<double>& yVec, const int M, const int N, CRNG& diffuse, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENC)
    {
        diffuse.map = getRandCRNG(MAP_LOWER_LIMIT, MAP_UPPER_LIMIT);
        diffuse.x = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        diffuse.y = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        if (diffuse.map == Chaos::LM)
        {
            diffuse.r = getRandDouble(R_LOWER_LIMIT, R_UPPER_LIMIT);
        }
        else if (diffuse.map == Chaos::SLMM)
        {
            diffuse.alpha = getRandDouble(ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT);
            diffuse.beta = getRandDouble(BETA_LOWER_LIMIT, BETA_UPPER_LIMIT);
        }
        else if (diffuse.map != Chaos::Arnold)
        {
            diffuse.myu = getRandDouble(MYU_LOWER_LIMIT, MYU_UPPER_LIMIT);
        }
        offset.diff_offset = getRandDouble(OFFSET_LOWER_LIMIT, OFFSET_UPPER_LIMIT);
        modifyParameters(diffuse, offset.diff_modifier, offset.diff_offset);
    }

    //Initiliaze Parameters

    Chaos map = diffuse.map;
    double x = diffuse.x;
    double y = diffuse.y;
    double alpha = diffuse.alpha;
    double beta = diffuse.beta;
    double myu = diffuse.myu;
    const double r = diffuse.r;

    auto start = steady_clock::now();

    for (int i = 0; i < N; ++i)
    {
        CRNGUpdate(x, y, alpha, beta, myu, r, map);
        xVec[i] = x;
        yVec[i] = y;
    }

    timeSince(start, "Diffusion CRNG");
}

// Top-level function to generate permutation vectors and run the permutation kernel
cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m)
{
    // Generate permutation vectors
    auto ptrU = getPermVec(dim[0], dim[1], permute[0], m);
    auto ptrV = getPermVec(dim[1], dim[0], permute[1], m);

    // Set grid and block data_size
    const dim3 grid(dim[0], dim[1], 1);
    const dim3 block(dim[2], 1, 1);

    // Call the kernel wrapper function
    Wrap_Permutation(d_img, d_imgtmp, ptrU, ptrV, grid, block, int(m));

    // Swap input and output vectors for subsequent rounds
    swap(d_img, d_imgtmp);
    return cudaDeviceSynchronize();
}

// Top-level function to generate diffusion vectors and run the diffusion kernel
cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], uint32_t diff_propfac, Mode m)
{
    // Initiliaze diffusion vectors    
    host_vector<double> randRowX(dim[1]), randRowY(dim[1]);

    // Generate diffusion vectors
    getDiffVecs(randRowX, randRowY, dim[0], dim[1], diffuse, m);

    // Transfer vectors from host to device
    device_vector<double> DRowX = randRowX, DRowY = randRowY;
    const double* rowXptr = (double*)(thrust::raw_pointer_cast(&DRowX[0]));
    const double* rowYptr = (double*)(thrust::raw_pointer_cast(&DRowY[0]));

    // Call the kernel wrapper function
    Wrap_Diffusion(d_img, d_imgtmp, rowXptr, rowYptr, dim, diffuse.r, int(m), diff_propfac);

    // Swap input and output vectors for subsequent rounds
    swap(d_img, d_imgtmp);

    return cudaDeviceSynchronize();
}

// Top-level function to calculate the sum of an image's pixels
cudaError_t CudaImageReduce(uint8_t* img, uint32_t& result, const int dim[])
{
    uint32_t* d_result;
    cudaMalloc(&d_result, sizeof(d_result));

    Wrap_ImageReduce(img, d_result, dim);

    cudaMemcpy(&result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return cudaDeviceSynchronize();
}

// Generates SHA256 hash of integer, adds up its bytes, divides the sum by 256 and returns its remainder.
static inline uint32_t getReducedHash(uint32_t value)
{
    long x = 0;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);

    auto buffer = (unsigned char*)calloc(SHA256_DIGEST_LENGTH, sizeof(unsigned char));

    // Extract each byte of <value> to buffer[]
    for (int i = 0; i < sizeof(value); ++i)
    {
        buffer[i] = (value >> (8 * i)) & 0xff;
    }

    SHA256_Update(&sha256, buffer, SHA256_DIGEST_LENGTH);
    SHA256_Final(hash, &sha256);

    // Convert SHA256 hash to a string
    stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
    {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    string hash_final = ss.str();

    // Calculate sum of the hash's characters
    for (int i = 0; i < hash_final.length(); ++i)
    {
        x += hash_final.at(i);
    }

    return (x % 256);
}

// Modifies all permutation & diffusion parameters using the Reverse Change Propagation offset
void reverseChangePropagation(std::vector<CRNG>& pVec, std::vector<CRNG>& dVec, uint32_t hash_sum_byte, Mode m)
{
    double offset = 0;
    CRNG permute;
    CRNG diffuse;

    if (m == Mode::ENC)
    {
        //Modifying all diffusion parameters
        for (int i = 0; i < dVec.size(); ++i)
        {
            diffuse = dVec[i];

            offset = getParameterOffset(hash_sum_byte);
            diffuse.x = diffuse.x + offset;
            diffuse.y = diffuse.y + offset;
            diffuse.alpha = diffuse.alpha + offset;
            diffuse.beta = diffuse.beta + offset;
            diffuse.myu = diffuse.myu + offset;
            diffuse.r = diffuse.r + offset;

            dVec[i] = diffuse;
        }

        //Modifying all permutation parameters
        for (int i = 0; i < pVec.size(); ++i)
        {
            permute = pVec[i];

            offset = getParameterOffset(hash_sum_byte);
            permute.x = permute.x + offset;
            permute.y = permute.y + offset;
            permute.r = permute.r + offset;
            permute.alpha = permute.alpha + offset;
            permute.beta = permute.beta + offset;
            permute.myu = permute.myu + offset;

            pVec[i] = permute;
        }
    }

    else if (m == Mode::DEC)
    {
        //Unmodifying all diffusion parameters
        for (int i = 0; i < dVec.size(); ++i)
        {
            diffuse = dVec[i];

            offset = getParameterOffset(hash_sum_byte);
            diffuse.x = diffuse.x - offset;
            diffuse.y = diffuse.y - offset;
            diffuse.alpha = diffuse.alpha - offset;
            diffuse.beta = diffuse.beta - offset;
            diffuse.myu = diffuse.myu - offset;
            diffuse.r = diffuse.r - offset;

            dVec[i] = diffuse;
        }

        //Unodifying all permutation parameters
        for (int i = 0; i < pVec.size(); ++i)
        {
            permute = pVec[i];

            offset = getParameterOffset(hash_sum_byte);
            permute.x = permute.x - offset;
            permute.y = permute.y - offset;
            permute.r = permute.r - offset;
            permute.alpha = permute.alpha - offset;
            permute.beta = permute.beta - offset;
            permute.myu = permute.myu - offset;

            pVec[i] = permute;
        }
    }
}

int Encrypt()
{
    Wrap_WarmUp();

    // Read the file and confirm it's been opened
    Mat img = imread(path.fn_img, cv::IMREAD_UNCHANGED);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };
    size_t img_bytes = (unsigned long long)dim[0] * (unsigned long long)dim[1] * (unsigned long long)dim[2] * sizeof(uint8_t);

    // Allocate VRAM for permutation vectors
    int* gpu_u, * gpu_v;
    cudaMalloc<int>(&gpu_v, dim[0] * sizeof(int));
    cudaMalloc<int>(&gpu_u, dim[1] * sizeof(int));

    // Allocate VRAM for input and output images
    uint8_t* d_img, * d_imgtmp;
    cudaMalloc<uint8_t>(&d_img, img_bytes);
    cudaMalloc<uint8_t>(&d_imgtmp, img_bytes);

    // Transfer image to VRAM
    cudaMemcpy(d_img, img.data, img_bytes, cudaMemcpyHostToDevice);

    // Variable to confirm success
    cudaError_t cudaStatus;

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------ENCRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n";

    // Calculate sum of the plain image's pixels
    uint32_t imageSum_plain = 0;
    cudaStatus = CudaImageReduce(d_img, imageSum_plain, dim);

    // Reduce the SHA256 hash of the imageSum
    auto start = steady_clock::now();
    uint32_t reducedHash_plain = getReducedHash(imageSum_plain);
    timeSince(start, "Generating reduced hash of plain image");

    // Propagation Factors to influence vector generation parameters
    prop.perm_propfac = reducedHash_plain ^ getRandUInt32(PERMUTE_PROPAGATION_LOWER_LIMIT, PERMUTE_PROPAGATION_UPPER_LIMIT);
    prop.diff_propfac = reducedHash_plain ^ getRandUInt32(DIFFUSE_PROPAGATION_LOWER_LIMIT, DIFFUSE_PROPAGATION_UPPER_LIMIT);

    // Parameter modifiers 
    offset.perm_modifier = getParameterOffset(prop.perm_propfac);
    offset.diff_modifier = getParameterOffset(prop.diff_propfac);

    start = steady_clock::now();

    // Encryption rounds
    cout << endl;
    for (int i = 0; i < config.rounds; i++)
    {
        cout << "\nX------ROUND " << i + 1 << "------X\n";


        /* Permute Image */
        for (int j = 0; j < config.rotations; j++)
        {
            cout << "\n     --Rotation " << j + 1 << "--     \n";

            cudaStatus = CudaPermute(d_img, d_imgtmp, dim, Mode::ENC);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "\nENC_Permutation Failed!";
                return -1;
            }
            pVec.push_back(permute[0]);
            pVec.push_back(permute[1]);
        }


        /*Diffuse image*/
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, prop.diff_propfac, Mode::ENC);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "\nENC_Diffusion Failed!";
            return -1;
        }

        dVec.push_back(diffuse);
    }

    timeSince(start, "Net Encryption Runtime");

    // Transfer cipherimage back to RAM and write it to disk
    cudaMemcpy(img.data, d_img, img_bytes, cudaMemcpyDeviceToHost);
    imwrite(path.fn_img_enc, img);

    // Calculate sum of the cipherimage's pixels
    uint32_t imageSum_ENC = 0;
    start = steady_clock::now();
    cudaStatus = CudaImageReduce(d_img, imageSum_ENC, dim);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "\nImage sum Failed!";
        return -1;
    }

    // Reduce the SHA256 hash of the imageSum
    start = steady_clock::now();
    uint32_t reducedHash_ENC = getReducedHash(imageSum_ENC);
    timeSince(start, "Generated reduced hash of cipherimage");

    //Modify parameters with Reverse Change Propagation Offset
    start = steady_clock::now();
    reverseChangePropagation(pVec, dVec, reducedHash_ENC, Mode::ENC);
    timeSince(start, "Reverse Change Propagation");

    if (PRINT_IMAGES == 1)
    {
        imshow("Encrypted Image", img);
    }

    // Calculate key size
    if (DEBUG_KEY == 1)
    {
        size_t key_size = CRNGVecSize(pVec) + CRNGVecSize(dVec);
        printf("\nNumber of rounds = %d", config.rounds);
        printf("\nKEY SIZE = %lu Bytes\n\n", key_size);
    }

    return 0;
}

int Decrypt()
{

    Wrap_WarmUp();

    // Read the file and confirm it's been opened
    cv::Mat img = cv::imread(path.fn_img_enc, cv::IMREAD_UNCHANGED);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };
    size_t img_bytes = (unsigned long long)dim[0] * (unsigned long long)dim[1] * (unsigned long long)dim[2] * sizeof(uint8_t);

    // Allocate VRAM for permutation vectors
    int* gpu_u, * gpu_v;
    cudaMalloc<int>(&gpu_v, dim[0] * sizeof(int));
    cudaMalloc<int>(&gpu_u, dim[1] * sizeof(int));

    // Allocate VRAM for input and output images
    uint8_t* d_img, * d_imgtmp;
    cudaMalloc<uint8_t>(&d_img, img_bytes);
    cudaMalloc<uint8_t>(&d_imgtmp, img_bytes);

    // Transfer cipherimage to VRAM
    cudaMemcpy(d_img, img.data, img_bytes, cudaMemcpyHostToDevice);

    // Variable to confirm success
    cudaError_t cudaStatus;

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------DECRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n";

    auto start = steady_clock::now();

    // Calculate sum of the cipherimage's pixels
    uint32_t imageSum_DEC = 0;
    cudaStatus = CudaImageReduce(d_img, imageSum_DEC, dim);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "\nImage Reduction Failed!";
        return -1;
    }

    // Reduce the SHA256 hash of the imageSum
    uint32_t reducedHash_DEC = getReducedHash(imageSum_DEC);

    // Recover parameters using Reverse Propagation Offset
    reverseChangePropagation(pVec, dVec, reducedHash_DEC, Mode::DEC);

    // Decryption rounds
    cout << endl;
    for (int i = config.rounds - 1; i >= 0; i--)
    {
        cout << "\nX------ROUND " << i + 1 << "------X\n";

        diffuse = dVec[i];

        //Undiffuse image 
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, prop.diff_propfac, Mode::DEC);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "DEC_Diffusion Failed!";
            return -1;
        }

        //Unpermute image
        for (int j = config.rotations - 1, idx = (2 * config.rotations) * i + 2 * j; j >= 0; j--, idx -= 2)
        {
            cout << "\n     --Rotation " << j + 1 << "--     \n";

            permute[0] = pVec[idx];
            permute[1] = pVec[idx + 1];


            cudaStatus = CudaPermute(d_img, d_imgtmp, dim, Mode::DEC);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "DEC_Permutation Failed!";
                return -1;
            }
        }
    }

    timeSince(start, "Gross Decryption Runtime");
    cout << endl;

    // Transfer decrypted image back to RAM and write it to disk
    cudaMemcpy(img.data, d_img, img_bytes, cudaMemcpyDeviceToHost);
    imwrite(path.fn_img_dec, img);

    // Display result
    if (PRINT_IMAGES == 1)
    {
        imshow("Decrypted Image", img);
    }

    return 0;
}

#endif

