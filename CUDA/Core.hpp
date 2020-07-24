#ifndef CORE_H
#define CORE_H

// Top-level encryption functions
#define DEBUG_VECTORS      0
#define DEBUG_PARAMETERS   0
#define DEBUG_KEY          1
#define DEBUG_PARAM_MOD    0
#define PRINT_IMAGES       0
#define DEBUG_CONSTRUCTORS 0

#define X_LOWER_LIMIT      0.1
#define X_UPPER_LIMIT      0.7
#define Y_LOWER_LIMIT      0.1
#define Y_UPPER_LIMIT      0.7
#define ALPHA_LOWER_LIMIT  0.905
#define ALPHA_UPPER_LIMIT  0.945
#define BETA_LOWER_LIMIT   2.97
#define BETA_UPPER_LIMIT   3.00
#define MYU_LOWER_LIMIT    0.40
#define MYU_UPPER_LIMIT    0.70
#define R_LOWER_LIMIT      1.15
#define R_UPPER_LIMIT      1.16
#define MAP_LOWER_LIMIT    1
#define MAP_UPPER_LIMIT    5
#define NONCE_LOWER_LIMIT  2400000
#define NONCE_UPPER_LIMIT  3600000  

#include <iostream> /*For IO*/
#include <cstdio>   /*For printf*/
#include <random>   /*For Mersenne Twister*/
#include <chrono>   /*For timing*/
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <openssl/sha.h> /*For sha256*/
#include <cstring>
#include <sstream>
#include <fstream>
#include "kernels.hpp"
#include "Classes.hpp"

using namespace cv;
using namespace std;
using namespace chrono;
using namespace thrust;
using namespace cub;

// Store generated parameters in vectors as File I/O is somehow proving to be the most trying part of this code
std::vector<Permuter> pVec;
std::vector<Diffuser> dVec;

//Sum of plain image used in diffusion kernels
uint32_t host_sum_plain = 0;

/*Function Prototypes*/
static inline void printImageContents(cv::Mat image,int channels);
void Initialize(std::string file, int rounds, int rotations_per_round);
double getParameterOffset(double value);
int* getPermVec(const int M, const int N, Permuter &permute, Mode m);

void getDiffVecs(host_vector<double> &xVec, host_vector<double> &yVec, const int M, const int N, Diffuser &diffuse, Mode m);
cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m);
cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, uint32_t host_sum_plain, const int dim[], Mode m);
cudaError_t CudaImageSumReduce(uint8_t *img, uint32_t *device_result, uint32_t &host_sum, const int dim[]);

static inline void getIntegerBytes(uint32_t value, unsigned char *&buffer);
static inline std::string sha256_hash_string (unsigned char hash[SHA256_DIGEST_LENGTH]);
static inline void calc_sha256(uint32_t value, uint32_t &hash_byte);
void hashParameters(std::vector<Permuter> &pVec, std::vector<Diffuser> &dVec, uint32_t hash_byte, Mode m);
int Encrypt(std::string file, int rounds, int rotations);
int Decrypt();
void printInt8Array(uint8_t *array, int length);
uint32_t getPowerOf10(uint32_t value);
size_t calculateKeySize(std::vector<Permuter> &pVec, std::vector<Diffuser> &dVec);

static inline void printImageContents(cv::Mat image,int channels)
{
  for(int i=0;i<image.rows;++i)
  { 
    printf("\n");
    for(int j=0;j<image.cols;++j)
    {
       for(int k=0; k < channels; ++k)
       {
         printf("%d\t",image.at<Vec3b>(i,j)[k]); 
       } 
     }
   }
   cout<<"\n";
}


// Initiliaze parameters
void Initialize(std::string file, int rounds, int rotations_per_round)
{
    config.rounds = rounds;
    config.rotations = rotations_per_round;
    path.buildPaths(file);
}


double getParameterOffset(uint32_t value)
{
  double parameter_offset = -0.1;
  int power = getPowerOf10(value);
  double val = (double)value;
  double divisor = pow(10, ((double)power + 2));
  
  if(value > 0)
  {
    parameter_offset = val / divisor;
    return parameter_offset;
  }
  
  else
  {
    cout<<"\nValue out of range\n";
    printf("\nvalue = %d", value);
    //parameter_offset = parameter_offset + 0.1;
    return parameter_offset;
  }
  
  return parameter_offset;
}


// Generate vector of N random numbers in [0,M]
int* getPermVec(const int M, const int N, Permuter &permute, Mode m)
{
    
    //Initiliaze CRNG
    if (m == Mode::ENCRYPT)
    {
        permute.x = randomNumber.getRandomDouble(X_LOWER_LIMIT , X_UPPER_LIMIT);
        permute.y = randomNumber.getRandomDouble(Y_LOWER_LIMIT , Y_UPPER_LIMIT);
        permute.x_bar = randomNumber.getRandomDouble(X_LOWER_LIMIT, X_UPPER_LIMIT);
        permute.y_bar = randomNumber.getRandomDouble(Y_LOWER_LIMIT, Y_UPPER_LIMIT);
        permute.alpha = randomNumber.getRandomDouble(ALPHA_LOWER_LIMIT , ALPHA_UPPER_LIMIT);
        permute.beta = randomNumber.getRandomDouble(BETA_LOWER_LIMIT , BETA_UPPER_LIMIT);
        permute.myu = randomNumber.getRandomDouble(MYU_LOWER_LIMIT , MYU_UPPER_LIMIT);
        permute.r = randomNumber.getRandomDouble(R_LOWER_LIMIT , R_UPPER_LIMIT);
        permute.map = randomNumber.crngAssigner(MAP_LOWER_LIMIT , MAP_UPPER_LIMIT);
    }
    //Initiliaze Parameters
    double x = permute.x;
    double y = permute.y;
    const double alpha = permute.alpha;
    const double beta = permute.beta;
    const double myu = permute.myu;
    const double r = permute.r;
    Chaos map = permute.map;
    
    CRNG crng(permute.x, permute.y, 0, 0, permute.alpha, permute.beta, permute.myu, permute.r, permute.map);
    
    host_vector<int> ranVec(N);
    const int exp = (int)pow(10, 9);

    //auto start = steady_clock::now();
    
    for(int i = 0; i < N ; ++i)
    {
        crng.CRNGUpdateHost(x, y, 0, 0, alpha, beta, myu, r, map);
        ranVec[i] = (int)(x * exp) % M;
    }
    
    if(DEBUG_VECTORS == 1)
    {
      cout<<"\nPERMUTATION VECTOR = ";
      for(int i = 0; i < N; ++i)
      {
        cout<<ranVec[i]<<" ";
      }
    }
    
    //cout << "\nPERMUTATION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
     
    device_vector<int> dVec = ranVec;
    return (int *)(thrust::raw_pointer_cast(&dVec[0]));
}

/* Generate 2 vectors of N random numbers in (0,1] each */
void getDiffVecs(host_vector<double> &xVec, host_vector<double> &yVec, const int M, const int N, Diffuser &diffuse, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENCRYPT)
    {
        diffuse.x = randomNumber.getRandomDouble(X_LOWER_LIMIT , X_UPPER_LIMIT);
        diffuse.y = randomNumber.getRandomDouble(Y_LOWER_LIMIT , Y_UPPER_LIMIT);
        diffuse.x_bar = randomNumber.getRandomDouble(X_LOWER_LIMIT, X_UPPER_LIMIT);
        diffuse.y_bar = randomNumber.getRandomDouble(Y_LOWER_LIMIT, Y_UPPER_LIMIT);
        diffuse.alpha = randomNumber.getRandomDouble(ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT);
        diffuse.beta = randomNumber.getRandomDouble(BETA_LOWER_LIMIT, BETA_UPPER_LIMIT);
        diffuse.myu = randomNumber.getRandomDouble(MYU_LOWER_LIMIT, MYU_UPPER_LIMIT);
        diffuse.r = randomNumber.getRandomDouble(R_LOWER_LIMIT , R_UPPER_LIMIT);
        diffuse.map = randomNumber.crngAssigner(1 , 5);
    }

    //Initiliaze Parameters
    
    double x = diffuse.x;
    double y = diffuse.y;
    double alpha = diffuse.alpha;
    double beta = diffuse.beta;
    double myu = diffuse.myu;
    const double r = diffuse.r;
    Chaos map = diffuse.map;
    
    CRNG crng(diffuse.x, diffuse.y, 0, 0, diffuse.alpha, diffuse.beta, diffuse.myu, diffuse.r, diffuse.map);
    
    //const int exp = (int)pow(10, 9);
    int i = 0;
    //auto start = steady_clock::now();
    
    for(i = 0; i < N; ++i)
    {
      crng.CRNGUpdateHost(x, y, 0, 0, alpha, beta, myu, r, map);
      xVec[i] = x;
      yVec[i] = y;
    }
    
    //cout << "DIFFUSION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
}

cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m)
{
    // Generate permutation vectors
    auto ptrU = getPermVec(dim[0], dim[1], permute[0], m);
    auto ptrV = getPermVec(dim[1], dim[0], permute[1], m);
    
    // Set grid and block data_size
    const dim3 grid(dim[0], dim[1], 1);
    const dim3 block(dim[2], 1, 1);

    //auto start = steady_clock::now();
    Wrap_RotatePerm(d_img, d_imgtmp, ptrU, ptrV, grid, block,int(m));
    swap(d_img, d_imgtmp);
    //cout << "Permutation: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";

    return cudaDeviceSynchronize();
}

cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, uint32_t host_sum_plain, const int dim[], Mode m)
{
    // Initiliaze diffusion vectors    
    host_vector<double> randRowX(dim[1]), randRowY(dim[1]);

    getDiffVecs(randRowX, randRowY, dim[0], dim[1], diffuse, m);

    device_vector<double> DRowX = randRowX, DRowY = randRowY;

    const double* rowXptr = (double*)(thrust::raw_pointer_cast(&DRowX[0]));
    const double* rowYptr = (double*)(thrust::raw_pointer_cast(&DRowY[0]));
        
    //auto start = steady_clock::now();
    Wrap_Diffusion(d_img, d_imgtmp, host_sum_plain, rowXptr, rowYptr, dim, diffuse.alpha, diffuse.beta, diffuse.myu, diffuse.r, int(m), int(diffuse.map));
    swap(d_img, d_imgtmp);
    //cout << "\nDiffusion: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";
    
    return cudaDeviceSynchronize();
}

cudaError_t CudaImageSumReduce(uint8_t *img, uint32_t *device_result, uint32_t &host_sum, const int dim[])
{
  Wrap_imageSumReduce(img, device_result, dim);
  cudaMemcpy(&host_sum, device_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  return cudaDeviceSynchronize(); 
}

static inline void getIntegerBytes(uint32_t value, unsigned char *&buffer)
{
   for(int i = 0; i < sizeof(value); ++i)
   {
     buffer[i] = (value >> (8 * i)) & 0xff;
   }
} 

static inline std::string sha256_hash_string (unsigned char hash[SHA256_DIGEST_LENGTH])
{
    
  stringstream ss;
  for(int i = 0; i < SHA256_DIGEST_LENGTH; i++)
  {
      ss << hex << setw(2) << setfill('0') << (int)hash[i];
  }
    
  return ss.str();
}

static inline void calc_sha256(uint32_t value, uint32_t &hash_byte)
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
  
  for(int i = 0; i < hash_final.length(); ++i)
  {
    x = x + hash_final.at(i);
  }
  
  x = x % 256;
  
  hash_byte = (uint8_t)x;
} 

void hashParameters(std::vector<Permuter> &pVec, std::vector<Diffuser> &dVec, uint32_t hash_byte, Mode m)
{
  double offset = 0;
  Permuter permute;
  Diffuser diffuse;
  
  if(m == Mode::ENCRYPT)
  {  
    /*Modifying all diffusion parameters*/
    for(int i = 0; i < dVec.size(); ++i)
    {
      diffuse = dVec[i];
      
      offset = getParameterOffset(hash_byte);
      diffuse.x = diffuse.x + offset;
      diffuse.y = diffuse.y + offset;
      diffuse.alpha = diffuse.alpha + offset;
      diffuse.beta = diffuse.beta + offset;
      diffuse.myu = diffuse.myu + offset;
      diffuse.r = diffuse.r + offset;
      
      dVec[i] = diffuse;
    }   
    
    /*Modifying all permutation parameters*/
    for(int i = 0; i < pVec.size(); ++i)
    {
      permute = pVec[i];
      
      offset = getParameterOffset(hash_byte);
      permute.x = permute.x + offset;
      permute.y = permute.y + offset;
      permute.r = permute.r + offset;
      permute.alpha = permute.alpha + offset;
      permute.beta = permute.beta + offset;
      permute.myu = permute.myu + offset;
      
      pVec[i] = permute;
    }
  }
  
  else if(m == Mode::DECRYPT)
  {
    /*Unmodifying all diffusion parameters*/
    for(int i = 0; i < dVec.size(); ++i)
    {
      diffuse = dVec[i];

      offset = getParameterOffset(hash_byte);
      diffuse.x = diffuse.x - offset;
      diffuse.y = diffuse.y - offset;
      diffuse.alpha = diffuse.alpha - offset;
      diffuse.beta = diffuse.beta - offset;
      diffuse.myu = diffuse.myu - offset;
      diffuse.r = diffuse.r - offset;
      
      dVec[i] = diffuse;  
    }
    
    /*Unodifying all permutation parameters*/
    for(int i = 0; i < pVec.size(); ++i)
    {
      permute = pVec[i];
      
      offset = getParameterOffset(hash_byte);
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

void printInt8Array(uint8_t *array, int length)
{
  for(int i = 0; i < length; ++i)
  {
    printf(" %d", array[i]);
  }
}

uint32_t getPowerOf10(uint32_t value)
{
  int power = 0;
  while(value > 0)
  {
    value = value / 10;
    ++power;
  }
  return power;
}

size_t calculateKeySize(std::vector<Permuter> &pVec, std::vector<Diffuser> &dVec)
{
  size_t key_size = 0;
  size_t unit = sizeof(double);
  Chaos map;
  size_t map_size = sizeof(map);
  for(Permuter permute : pVec)
  {
    if(int(permute.map) == 1)
    {
      key_size = key_size + (unit * 2) + map_size;
    }
    
    else if(int(permute.map) == 3 || int(permute.map) == 5 )
    {
      key_size = key_size + (unit * 4) + map_size;
    }
    
    else if(int(permute.map) == 2 || int(permute.map) == 4)
    {
      key_size = key_size + (unit * 3) + map_size;
    }
  }
  
  for(Diffuser diffuse : dVec)
  {
    if(int(diffuse.map) == 1)
    {
      key_size = key_size + (unit * 2) + map_size;
    }
    
    else if(int(diffuse.map) == 3 || int(diffuse.map) == 5 )
    {
      key_size = key_size + (unit * 4) + map_size;
    }
    
    else if(int(diffuse.map) == 2 || int(diffuse.map) == 4)
    {
      key_size = key_size + (unit * 3) + map_size;
    }
  }
  
  key_size = key_size + sizeof(uint32_t);
  return key_size;
}

int Encrypt(std::string file, int rounds, int rotations)
{
    kernel_WarmUp();
    Initialize(file, rounds, rotations);

    // Read the file and confirm it's been opened
    auto start_read = steady_clock::now();
    Mat img = imread(path.fn_img, cv::IMREAD_UNCHANGED);
    auto end_read = steady_clock::now();
    auto duration_read = (int)duration_cast<milliseconds>(end_read - start_read).count();
    printf("\nRead plain image = %d ms", duration_read); 
    
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
    
 
    //Resize image
    //cv::resize(img, img, cv::Size(4 , 4));
    
    
    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };
    
    // Printing image
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nOriginal image \n";
      printImageContents(img, dim[2]);
    }
    
    uint8_t* d_img, * d_imgtmp;

    uint32_t host_sum_ENC = 0;
    uint32_t *device_sum_ENC, *device_sum_plain, *device_result;
    
    size_t device_sum_size = sizeof(device_sum_ENC);
    
    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
    
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
    
    int *gpu_u;
    int *gpu_v;
    
    cudaError_t cudaStatus;
    
    uint32_t hash_byte = 0; 
    
    cudaMalloc(&device_sum_plain, device_sum_size);
    cudaMalloc(&device_sum_ENC, device_sum_size);
    cudaMalloc(&device_result, device_sum_size);
    
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);
    
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);
    
    // Upload image to device
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);
    
    // Calculating the sum of plain image, the sum of whose hash will be used in diffusion for forward propagation
    
    auto start_sumplain = steady_clock::now();
    cudaStatus = CudaImageSumReduce(d_img, device_sum_plain, host_sum_plain, dim);
    auto end_sumplain = steady_clock::now();
    auto duration_sumplain = (int)duration_cast<microseconds>(end_sumplain - start_sumplain).count();
    printf("\nSum of plain image in host = %d us", duration_sumplain);
    
    if(cudaStatus != cudaSuccess)
    {
      cerr << "\nimage sum Failed!";
      cout<<"\nimage sum kernel error / status = "<<cudaStatus;
      return -1;
    }
    
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\nhost_sum_plain = %d", host_sum_plain);
      printf("\nhash_byte = %d", hash_byte);
    }
    
    // Show original image
    //imshow("Original", img);

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

            cudaStatus = CudaPermute(d_img, d_imgtmp, dim, Mode::ENCRYPT);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "\nENC_Permutation Failed!";
                cout<<"\nENC_Permutation kernel error / status = "<<cudaStatus;
                return -1;
            }
            
            
            pVec.push_back(permute[0]);
            pVec.push_back(permute[1]);
        }
        
        
        /*Diffuse image*/
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, host_sum_plain, dim, Mode::ENCRYPT);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "\nENC_Diffusion Failed!";
            cout<<"\nENC_Diffusion kernel error / status = "<<cudaStatus;
            return -1;
        }
       
        dVec.push_back(diffuse);
  }
    
    auto end_enc = steady_clock::now();
    auto duration_enc = (int)duration_cast<milliseconds>(end_enc - start_enc).count();
    printf("\nDURATION OF ENCRYPTION = %d ms", duration_enc);
    
    
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    
    auto start_write = steady_clock::now();
    imwrite(path.fn_img_enc, img);
    auto end_write = steady_clock::now();
    auto duration_write = (int)duration_cast<milliseconds>(end_write - start_write).count();
    printf("\nWrite encrypted image = %d ms", duration_write); 
    
    auto start_imagesum = steady_clock::now();    
    /*Calculate sum of encrypted image*/
    cudaStatus = CudaImageSumReduce(d_img, device_result, host_sum_ENC, dim);
    auto end_imagesum = steady_clock::now();
    auto duration_imagesum = (int)duration_cast<microseconds>(end_imagesum - start_imagesum).count();
    printf("\nSum of encrypted image in hhost = %d us", duration_imagesum);
  
    if (cudaStatus != cudaSuccess)
    {
      cerr << "\nimage sum Failed!";
      cout<<"\nimage sum kernel error / status = "<<cudaStatus;
      return -1;
    }

    auto start_hash = steady_clock::now(); 
    //Calculate sha256 hash of the sum of the encrypted image
    calc_sha256(host_sum_ENC, hash_byte);
    auto end_hash = steady_clock::now();
    auto duration_hash = (int)duration_cast<microseconds>(end_hash - start_hash).count();
    printf("\nCompute sha256 hash of the sum of the encrypted image = %d us", duration_hash);
   
    auto start_modification = steady_clock::now(); 
    //Modify the parameters using an offset generated from the sum of the hash
    hashParameters(pVec, dVec, hash_byte, Mode::ENCRYPT);
    auto end_modification = steady_clock::now();
    auto duration_modification = (int)duration_cast<microseconds>(end_modification - start_modification).count();
    printf("\nModify the parameters using the offset of the generated hash = %d us", duration_modification);
    
    if(DEBUG_PARAMETERS == 1)
    {    
      printf("\nhost_sum_ENC = %d", host_sum_ENC);
      printf("\nhash_byte = %d\n", hash_byte);
    }
    
    // Display encrypted image 
    //imshow("Encrypted", img);
    
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nEncrypted image \n";
      printImageContents(img, dim[2]);
    }
    
    //cudaDeviceReset();
    
    //Calculating the size of the key
    if(DEBUG_KEY == 1)
    {
      size_t key_size = calculateKeySize(pVec, dVec);
      printf("\nNumber of rounds = %d", config.rounds);
      printf("\nKEY SIZE = %lu Bytes", key_size);
    }
    
    return 0;
}

int Decrypt()
{
    
    kernel_WarmUp();

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
    
    //img.at<Vec3b>(0, 0)[0] = img.at<Vec3b>(0, 0)[0] + 1;
    //Resize image
    //cv::resize(img, img, cv::Size(4 , 4));
    
    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };
    
    //Printing image
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nCipher image \n";
      printImageContents(img, dim[2]);
    }
    
    // Upload image and LUTs to device
    uint8_t* d_img, * d_imgtmp;
    
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
    
    int *gpu_u;
    int *gpu_v;
    
    uint32_t hash_byte = 209;
    
    uint32_t host_sum_DEC = 0;
    uint32_t *device_sum_DEC, *device_sum_plain, *device_result;
   
    size_t device_sum_size = sizeof(device_sum_DEC);
    
    cudaError_t cudaStatus;
    
    cudaMalloc(&device_sum_DEC, device_sum_size);
    cudaMalloc(&device_sum_plain, device_sum_size);
    cudaMalloc(&device_result, device_sum_size);
    
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);

    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
        
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);
    
    cudaMemcpy(device_sum_plain, &host_sum_plain, device_sum_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);
    
    //Calculate sum of encrypted image
    cudaStatus = CudaImageSumReduce(d_img, device_result, host_sum_DEC, dim);
    
    if (cudaStatus != cudaSuccess)
    {
      cerr << "\nimage sum Failed!";
      cout<<"\nimage sum kernel error / status = "<<cudaStatus;
      return -1;
    }
    
    auto start_hash = steady_clock::now();
    //Calculate sum of hash of sum of encrypted image
    calc_sha256(host_sum_DEC, hash_byte);
    auto end_hash = steady_clock::now();
    auto duration_hash = (int)duration_cast<microseconds>(end_hash - start_hash).count();
    printf("\nCompute sha256 hash of the sum of the encrypted image = %d us", duration_hash);
    
    auto start_modification = steady_clock::now();
    //Modify the parameters using an offset generated from the sum of the hash
    hashParameters(pVec, dVec, hash_byte, Mode::DECRYPT);
    auto end_modification = steady_clock::now();
    auto duration_modification = (int)duration_cast<microseconds>(end_modification - start_modification).count();
    printf("\nModify the parameters using the offset of the generated hash = %d us", duration_modification);
    
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\nhost_sum_DEC = %d", host_sum_DEC);
      printf("\nhash_byte = %d\n", hash_byte);
    }

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
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, host_sum_plain, dim, Mode::DECRYPT);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "DEC_Diffusion Failed!";
            cout<<"\nDEC_DIffusion kernel error / status = "<<cudaStatus;
            return -1;
        }    
        
        //Unpermute image
        for (int j = config.rotations - 1, idx = (2 * config.rotations) * i + 2 * j; j >= 0; j--, idx-=2)
        {
            cout << "\n     --Rotation " << j + 1 << "--     \n";
            
            permute[0] = pVec[idx];
            permute[1] = pVec[idx + 1];
            
            
            cudaStatus = CudaPermute(d_img, d_imgtmp, dim, Mode::DECRYPT);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "DEC_Permutation Failed!";
                cout<<"\nDEC_Permutation kernel error / status = "<<cudaStatus;
                return -1;
            }
        }
    }
    auto end_dec = steady_clock::now();
    auto duration_dec = (int)duration_cast<milliseconds>(end_dec - start_dec).count();
    printf("\nDURATION OF DECRYPTION = %d ms", duration_dec);
    
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    
    auto start_write = steady_clock::now();
    imwrite(path.fn_img_dec, img);
    auto end_write = steady_clock::now();
    auto duration_write = (int)duration_cast<milliseconds>(end_write - start_write).count();
    printf("\nWrite decrypted image = %d ms", duration_write);
    
    // Display decrypted image
    //imshow("Decrypted", img);
   
    //cudaDeviceReset();
    //waitKey(0);
    
    //Printing image
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nDecrypted image \n";
      printImageContents(img, dim[2]);
    }
     
    return 0;
}

#endif
