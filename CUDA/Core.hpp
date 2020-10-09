#ifndef CORE_H
#define CORE_H

// Top-level encryption functions

#define DEBUG_KEY                        1
#define PRINT_IMAGES                     1

#define XY_LOWER_LIMIT                    0.1
#define XY_UPPER_LIMIT                    0.7
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
#define OFFSET_LOWER_LIMIT          0.00001  
#define OFFSET_UPPER_LIMIT          0.00002

#include <cstdio>   //printf
#include <chrono>   //Timing
#include <openssl/sha.h> //sha256
#include <opencv2/opencv.hpp> //Convenient Image I/O
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thrust/host_vector.h> // Convenient vector transfer between host and device
#include <thrust/device_vector.h>
#include "Classes.hpp"
#include "functions.hpp"

using namespace cv;
using namespace std;
using namespace chrono;
using namespace thrust;

// Store generated parameters in vectors instead of writing to disk as File I/O is somehow proving to be the most trying part of this code
std::vector<CRNG> pVec, dVec;

// Sum of plain image
uint32_t host_hash_sum_plain = 0;
uint32_t hash_sum_byte_plain = 0;

/*Function Prototypes*/

void modifyPermutationParameters(CRNG& permute, double permutation_parameter_modifier, double perm_param_offset);
void modifyDiffusionParameters(CRNG& diffuse, double diffusion_parameter_modifier, double diff_param_offset);

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
}

// Returns number of digits in provided value
uint32_t getPowerOf10(uint32_t value)
{
    int power = 0;
    while (value > 0)
    {
        value = value / 10;
        ++power;
    }
    return power;
}

double getParameterOffset(uint32_t value)
{
    double parameter_offset = -0.1;
    int power = getPowerOf10(value);
    double val = (double)value;
    double divisor = pow(10, ((double)power + 2));

    if (value > 0)
    {
        parameter_offset = val / divisor;
        return parameter_offset;
    }

    else
    {
        cout << "\nValue out of range\n";
        printf("\nvalue = %d", value);
        //parameter_offset = parameter_offset + 0.1;
        return parameter_offset;
    }

    return parameter_offset;
}

// Generates vector of N random numbers in [0,M]
int* getPermVec(const int M, const int N, CRNG& permute, Mode m)
{

    //Initiliaze CRNG
    if (m == Mode::ENC)
    {
        permute.x = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        permute.y = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        permute.alpha = getRandDouble(ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT);
        permute.beta = getRandDouble(BETA_LOWER_LIMIT, BETA_UPPER_LIMIT);
        permute.myu = getRandDouble(MYU_LOWER_LIMIT, MYU_UPPER_LIMIT);
        permute.r = getRandDouble(R_LOWER_LIMIT, R_UPPER_LIMIT);
        permute.map = getRandCRNG(MAP_LOWER_LIMIT, MAP_UPPER_LIMIT);
        offset.permute_param_offset = getRandDouble(OFFSET_LOWER_LIMIT, OFFSET_UPPER_LIMIT);
        modifyPermutationParameters(permute, offset.permute_param_modifier, offset.permute_param_offset);
    }

    //Initiliaze Parameters
    double x = permute.x;
    double y = permute.y;
    const double alpha = permute.alpha;
    const double beta = permute.beta;
    const double myu = permute.myu;
    const double r = permute.r;
    Chaos map = permute.map;

    host_vector<int> ranVec(N);
    const int exp = (int)pow(10, 9);

    auto start = steady_clock::now();

    for (int i = 0; i < N; ++i)
    {
        CRNGUpdate(x, y, alpha, beta, myu, r, map);
        ranVec[i] = (int)(x * exp) % M;
    }

    cout << "\nPERMUTATION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";

    device_vector<int> dVec = ranVec;
    return (int*)(thrust::raw_pointer_cast(&dVec[0]));
}

// Generates vector of M random numbers in (0.0,1.0)
void getDiffVecs(host_vector<double>& xVec, host_vector<double>& yVec, const int M, const int N, CRNG& diffuse, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENC)
    {
        diffuse.x = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        diffuse.y = getRandDouble(XY_LOWER_LIMIT, XY_UPPER_LIMIT);
        diffuse.alpha = getRandDouble(ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT);
        diffuse.beta = getRandDouble(BETA_LOWER_LIMIT, BETA_UPPER_LIMIT);
        diffuse.myu = getRandDouble(MYU_LOWER_LIMIT, MYU_UPPER_LIMIT);
        diffuse.r = getRandDouble(R_LOWER_LIMIT, R_UPPER_LIMIT);
        diffuse.map = getRandCRNG(MAP_LOWER_LIMIT, MAP_UPPER_LIMIT);
        offset.diffuse_param_offset = getRandDouble(OFFSET_LOWER_LIMIT, OFFSET_UPPER_LIMIT);
        modifyDiffusionParameters(diffuse, offset.diffuse_param_modifier, offset.diffuse_param_offset);
    }

    //Initiliaze Parameters

    double x = diffuse.x;
    double y = diffuse.y;
    double alpha = diffuse.alpha;
    double beta = diffuse.beta;
    double myu = diffuse.myu;
    const double r = diffuse.r;
    Chaos map = diffuse.map;

    int i = 0;
    auto start = steady_clock::now();

    for (i = 0; i < N; ++i)
    {
        CRNGUpdate(x, y, alpha, beta, myu, r, map);
        xVec[i] = x;
        yVec[i] = y;
    }

    cout << "DIFFUSION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
}

// Modifies permutation parameters for a given permutation round
void modifyPermutationParameters(CRNG& permute, double permutation_parameter_modifier, double perm_param_offset)
{
    switch (int(permute.map))
    {
        case 1:
        {
            permute.x = permute.x + permutation_parameter_modifier + perm_param_offset;
            permute.y = permute.y + permutation_parameter_modifier + perm_param_offset;
        }
        break;

        case 2:
        {
            permute.x = permute.x + permutation_parameter_modifier + perm_param_offset;
            permute.y = permute.y + permutation_parameter_modifier + perm_param_offset;
            permute.r = permute.r + permutation_parameter_modifier + perm_param_offset;
        }
        break;

        case 3:
        {
            permute.x = permute.x + permutation_parameter_modifier + perm_param_offset;
            permute.y = permute.y + permutation_parameter_modifier + perm_param_offset;
            permute.alpha = permute.alpha + permutation_parameter_modifier + perm_param_offset;
            permute.beta = permute.beta + permutation_parameter_modifier + perm_param_offset;
        }
        break;

        case 4:
        {
            permute.x = permute.x + permutation_parameter_modifier + perm_param_offset;
            permute.y = permute.y + permutation_parameter_modifier + perm_param_offset;
            permute.myu = permute.myu + permutation_parameter_modifier + perm_param_offset;
        }
        break;

        case 5:
        {
            permute.x = permute.x + permutation_parameter_modifier + perm_param_offset;
            permute.y = permute.y + permutation_parameter_modifier + perm_param_offset;
            permute.myu = permute.myu + permutation_parameter_modifier + perm_param_offset;
        }
        break;

        default:cout << "\nInvalid map choice for permutation\n";
    }
}

// Modifies permutation parameters for a given encryption round
void modifyDiffusionParameters(CRNG& diffuse, double diffusion_parameter_modifier, double diff_param_offset)
{
    switch (int(diffuse.map))
    {
        case 1:
        {
            diffuse.x = diffuse.x + diffusion_parameter_modifier + diff_param_offset;
            diffuse.y = diffuse.y + diffusion_parameter_modifier + diff_param_offset;
        }
        break;

        case 2:
        {
            diffuse.x = diffuse.x + diffusion_parameter_modifier + diff_param_offset;
            diffuse.y = diffuse.y + diffusion_parameter_modifier + diff_param_offset;
            diffuse.r = diffuse.r + diffusion_parameter_modifier + diff_param_offset;
        }
        break;

        case 3:
        {
            diffuse.x = diffuse.x + diffusion_parameter_modifier + diff_param_offset;
            diffuse.y = diffuse.y + diffusion_parameter_modifier + diff_param_offset;
            diffuse.alpha = diffuse.alpha + diffusion_parameter_modifier + diff_param_offset;
            diffuse.beta = diffuse.beta + diffusion_parameter_modifier + diff_param_offset;
        }
        break;

        case 4:
        {
            diffuse.x = diffuse.x + diffusion_parameter_modifier + diff_param_offset;
            diffuse.y = diffuse.y + diffusion_parameter_modifier + diff_param_offset;
            diffuse.myu = diffuse.myu + diffusion_parameter_modifier + diff_param_offset;
        }
        break;

        case 5:
        {
            diffuse.x = diffuse.x + diffusion_parameter_modifier + diff_param_offset;
            diffuse.y = diffuse.y + diffusion_parameter_modifier + diff_param_offset;
            diffuse.myu = diffuse.myu + diffusion_parameter_modifier + diff_param_offset;
        }
        break;

        default:cout << "\nInvalid map choice for permutation\n";
    }
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

    //auto start = steady_clock::now();
    // Call the kernel wrapper function
    Wrap_Permutation(d_img, d_imgtmp, ptrU, ptrV, grid, block, int(m));
    // Swap input and output vectors for subsequent rounds
    swap(d_img, d_imgtmp);
    //cout << "Permutation: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";

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

    //auto start = steady_clock::now();
    // Call the kernel wrapper function
    Wrap_Diffusion(d_img, d_imgtmp, rowXptr, rowYptr, dim, diffuse.r, int(m), diff_propfac);
    // Swap input and output vectors for subsequent rounds
    swap(d_img, d_imgtmp);
    //cout << "\nDiffusion: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";

    return cudaDeviceSynchronize();
}

// Top-level function for calculating sum of an image's pixels
cudaError_t CudaImageSumReduce(uint8_t* img, uint32_t* device_result, uint32_t& host_sum, const int dim[])
{
    Wrap_ImageReduce(img, device_result, dim);
    cudaMemcpy(&host_sum, device_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return cudaDeviceSynchronize();
}

// Extract each byte of an integer to a character array
static inline void getIntegerBytes(uint32_t value, unsigned char*& buffer)
{
    for (int i = 0; i < sizeof(value); ++i)
    {
        buffer[i] = (value >> (8 * i)) & 0xff;
    }
}

// Converts SHA256 hash to std::string
static inline std::string sha256_hash_string(unsigned char hash[SHA256_DIGEST_LENGTH])
{
    stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
    {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    return ss.str();
}

// Generates SHA256 hash of integer, adds up its bytes, divides the sum by 256 and returns its remainder.
static inline void calc_sum_of_hash(uint32_t value, uint32_t& hash_sum_byte)
{
    long x = 0;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);

    const int bufSize = SHA256_DIGEST_LENGTH;
    //const int bufSize = 3;
    unsigned char* buffer = (unsigned char*)calloc(bufSize, sizeof(unsigned char));

    getIntegerBytes(value, buffer);
    //cout <<"\nEnter buffer ";
    //cin >> buffer;
    SHA256_Update(&sha256, buffer, bufSize);

    SHA256_Final(hash, &sha256);

    std::string hash_final = sha256_hash_string(hash);

    for (int i = 0; i < hash_final.length(); ++i)
    {
        x = x + hash_final.at(i);
    }

    x = x % 256;

    hash_sum_byte = (uint8_t)x;
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

    uint8_t* d_img, * d_imgtmp;

    uint32_t host_hash_sum_ENC = 0;
    uint32_t* device_hash_sum_ENC, * device_hash_sum_plain;

    size_t device_hash_sum_size = sizeof(device_hash_sum_ENC);

    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);

    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);

    int* gpu_u;
    int* gpu_v;

    cudaError_t cudaStatus;

    uint32_t hash_sum_byte_ENC = 0;

    //Allocating device memory for sums of hash
    cudaMalloc(&device_hash_sum_plain, device_hash_sum_size);
    cudaMalloc(&device_hash_sum_ENC, device_hash_sum_size);


    //Allocating device memory for permutation vectors
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);

    //Allocating device memory for input and output images
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);

    // Upload image to device
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);

    // Calculating the sum of plain image 
    auto start_sumplain = steady_clock::now();
    cudaStatus = CudaImageSumReduce(d_img, device_hash_sum_plain, host_hash_sum_plain, dim);
    auto end_sumplain = steady_clock::now();
    auto duration_sumplain = (int)duration_cast<microseconds>(end_sumplain - start_sumplain).count();
    printf("\nSum of plain image in host = %d us", duration_sumplain);

    //Calculate sum of the sha256 hash of the sum of the plain image
    calc_sum_of_hash(host_hash_sum_plain, hash_sum_byte_plain);

    //Factor to induce propagation in permutation vector generation parameters
    propagator.perm_propfac = hash_sum_byte_plain ^ getRandUInt32(PERMUTE_PROPAGATION_LOWER_LIMIT, PERMUTE_PROPAGATION_UPPER_LIMIT);
    //Factor to induce forward propagation in diffusion vector generation parameters and diffusion kernel
    propagator.diff_propfac = hash_sum_byte_plain ^ getRandUInt32(DIFFUSE_PROPAGATION_LOWER_LIMIT, DIFFUSE_PROPAGATION_UPPER_LIMIT);

    //Permutation and diffusion parameter modifiers 
    offset.permute_param_modifier = getParameterOffset(propagator.perm_propfac);
    offset.diffuse_param_modifier = getParameterOffset(propagator.diff_propfac);

    if (cudaStatus != cudaSuccess)
    {
        cerr << "\nimage sum Failed!";
        cout << "\nimage sum kernel error / status = " << cudaStatus;
        return -1;
    }

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------ENCRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n\n";

    auto start_enc = steady_clock::now();

    // Encryption rounds
    for (int i = 0; i < config.rounds; i++)
    {
        cout << "X------ROUND " << i + 1 << "------X\n";


        /* Permute Image */
        for (int j = 0; j < config.rotations; j++)
        {
            cout << "\n     --Rotation " << j + 1 << "--     \n";

            cudaStatus = CudaPermute(d_img, d_imgtmp, dim, Mode::ENC);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "\nENC_Permutation Failed!";
                cout << "\nENC_Permutation kernel error / status = " << cudaStatus;
                return -1;
            }


            pVec.push_back(permute[0]);
            pVec.push_back(permute[1]);
        }


        /*Diffuse image*/
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, propagator.diff_propfac, Mode::ENC);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "\nENC_Diffusion Failed!";
            cout << "\nENC_Diffusion kernel error / status = " << cudaStatus;
            return -1;
        }

        dVec.push_back(diffuse);
    }

    auto end_enc = steady_clock::now();
    auto duration_enc = (int)duration_cast<milliseconds>(end_enc - start_enc).count();
    printf("\nDURATION OF ENCRYPTION = %d ms", duration_enc);

    //Copy encrypted image from device memory to host memory 
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);

    //Write encrypted image to disk
    auto start_write = steady_clock::now();
    imwrite(path.fn_img_enc, img);
    auto end_write = steady_clock::now();
    auto duration_write = (int)duration_cast<milliseconds>(end_write - start_write).count();
    printf("\nWrite encrypted image = %d ms", duration_write);

    auto start_imagesum = steady_clock::now();
    //Calculate sum of encrypted image
    cudaStatus = CudaImageSumReduce(d_img, device_hash_sum_ENC, host_hash_sum_ENC, dim);
    auto end_imagesum = steady_clock::now();
    auto duration_imagesum = (int)duration_cast<microseconds>(end_imagesum - start_imagesum).count();
    printf("\nSum of encrypted image in hhost = %d us", duration_imagesum);

    if (cudaStatus != cudaSuccess)
    {
        cerr << "\nimage sum Failed!";
        cout << "\nimage sum kernel error / status = " << cudaStatus;
        return -1;
    }

    auto start_hash = steady_clock::now();
    //Calculate sum of the sha256 hash of the sum of the encrypted image
    calc_sum_of_hash(host_hash_sum_ENC, hash_sum_byte_ENC);
    auto end_hash = steady_clock::now();
    auto duration_hash = (int)duration_cast<microseconds>(end_hash - start_hash).count();
    printf("\nCompute sha256 hash of the sum of the encrypted image = %d us", duration_hash);

    auto start_modification = steady_clock::now();
    //Modify the parameters by adding Reverse Change Propagation Offset to them
    reverseChangePropagation(pVec, dVec, hash_sum_byte_ENC, Mode::ENC);
    auto end_modification = steady_clock::now();
    auto duration_modification = (int)duration_cast<microseconds>(end_modification - start_modification).count();
    printf("\nModify the parameters using the offset of the generated hash = %d us", duration_modification);

    if (PRINT_IMAGES == 1)
    {
        namedWindow("enc");
        imshow("enc", img);
    }

    //Calculating the size of the key
    if (DEBUG_KEY == 1)
    {
        size_t key_size = (sizeof(Chaos) * (pVec.size() + dVec.size())) + CRNGVecSize(pVec) + CRNGVecSize(dVec) + (sizeof(propagator.perm_propfac) * 2);
        printf("\nNumber of rounds = %d", config.rounds);
        printf("\nKEY SIZE = %lu Bytes", key_size);
    }

    return 0;
}

int Decrypt()
{

    Wrap_WarmUp();

    // Read the file and confirm it's been opened
    auto start_read = steady_clock::now();
    cv::Mat img = cv::imread(path.fn_img_enc, cv::IMREAD_UNCHANGED);
    auto end_read = steady_clock::now();
    auto duration_read = (int)duration_cast<milliseconds>(end_read - start_read).count();
    printf("\nRead the encrypted image = %d ms", duration_read);

    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };


    // Upload image and LUTs to device
    uint8_t* d_img, * d_imgtmp;

    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);

    int* gpu_u;
    int* gpu_v;

    uint32_t hash_sum_byte_DEC = 0;

    uint32_t host_hash_sum_DEC = 0;
    uint32_t* device_hash_sum_DEC;

    size_t device_hash_sum_size = sizeof(device_hash_sum_DEC);

    cudaError_t cudaStatus;

    //Allocating device memory for sum of plain image and encrypted image
    cudaMalloc(&device_hash_sum_DEC, device_hash_sum_size);


    //Allocating device memory for permutation vectors
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);

    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);

    //Allocating device memory for encrypted image and decrypted image      
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);

    //Copying encrypted image from host memory to device memory
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);

    //Calculate sum of encrypted image
    cudaStatus = CudaImageSumReduce(d_img, device_hash_sum_DEC, host_hash_sum_DEC, dim);

    if (cudaStatus != cudaSuccess)
    {
        cerr << "\nimage sum Failed!";
        cout << "\nimage sum kernel error / status = " << cudaStatus;
        return -1;
    }

    auto start_hash = steady_clock::now();
    //Calculate sum of hash of sum of encrypted image
    calc_sum_of_hash(host_hash_sum_DEC, hash_sum_byte_DEC);
    auto end_hash = steady_clock::now();
    auto duration_hash = (int)duration_cast<microseconds>(end_hash - start_hash).count();
    printf("\nCompute sha256 hash of the sum of the encrypted image = %d us", duration_hash);

    auto start_modification = steady_clock::now();
    //Recover all permutation and diffusion parameters by subtracting from said parameters, the Reverse Propagation Offset
    reverseChangePropagation(pVec, dVec, hash_sum_byte_DEC, Mode::DEC);
    auto end_modification = steady_clock::now();
    auto duration_modification = (int)duration_cast<microseconds>(end_modification - start_modification).count();
    printf("\nModify the parameters using the offset of the generated hash = %d us", duration_modification);

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------DECRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n\n";

    auto start_dec = steady_clock::now();

    //Decryption rounds
    for (int i = config.rounds - 1; i >= 0; i--)
    {
        cout << "X------ROUND " << i + 1 << "------X\n";

        diffuse = dVec[i];

        //Undiffuse image 
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, propagator.diff_propfac, Mode::DEC);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "DEC_Diffusion Failed!";
            cout << "\nDEC_DIffusion kernel error / status = " << cudaStatus;
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
                cout << "\nDEC_Permutation kernel error / status = " << cudaStatus;
                return -1;
            }
        }
    }
    auto end_dec = steady_clock::now();
    auto duration_dec = (int)duration_cast<milliseconds>(end_dec - start_dec).count();
    printf("\nDURATION OF DECRYPTION = %d ms", duration_dec);

    //Copy decrypted image from device memory to host memory
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);

    //Write decrypted image to disk
    auto start_write = steady_clock::now(); 
    imwrite(path.fn_img_dec, img);
    auto end_write = steady_clock::now();
    auto duration_write = (int)duration_cast<milliseconds>(end_write - start_write).count();
    printf("\nWrite decrypted image = %d ms", duration_write);

    //Printing image
    if (PRINT_IMAGES == 1)
    {
        namedWindow("dec");
        imshow("dec", img);
    }

    return 0;
}

#endif

