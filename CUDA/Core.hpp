#ifndef CORE_H
#define CORE_H

// Top-level encryption functions
#define DEBUG_VECTORS      0
#define DEBUG_PARAMETERS   0
#define PRINT_IMAGES       0
#define DEBUG_CONSTRUCTORS 0
#define EXP                1000000000

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
#include <sstream>
#include <fstream>
#include "kernels.hpp"
#include "Classes.hpp"

using namespace cv;
using namespace std;
using namespace chrono;
using namespace thrust;

// Store generated parameters in vectors as File I/O is somehow proving to be the most trying part of this code
std::vector<Permuter> pVec;
std::vector<Diffuser> dVec;

// Temporarily store generated parameters at the end of each round
std::vector<Permuter> pVecTemp;
std::vector<Diffuser> dVecTemp;

static inline void getIntegerBytes(uint32_t value, char *&buffer)
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

static inline const char* calc_sha256(uint32_t value)
{    
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
    
  const int bufSize = SHA256_DIGEST_LENGTH;
  //const int bufSize = 3;
  char* buffer = (char*)calloc(bufSize, sizeof(char));
  int bytesRead = 0;
    
  getIntegerBytes(value, buffer);
  //cout <<"\nEnter buffer ";
  //cin >> buffer;
  SHA256_Update(&sha256, buffer, bufSize);
    
  SHA256_Final(hash, &sha256);

  std::string hash_final = sha256_hash_string(hash);
    
  return hash_final.c_str();
} 

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
void Initialize()
{
    string file;
    cout << "Filename: ";
    cin >> file;
    cout << "Rounds: ";
    //config.rounds = 2;
    scanf("%d",&config.rounds);
    //config.rounds = (config.rounds > 0) ? config.rounds : 2;
    config.rotations = 2;
    path.buildPaths(file);
}

// Generate vector of N random numbers in [0,M]
int* getPermVec(const int M, const int N, Permuter &permute, Mode m)
{
    
    //Initiliaze CRNG
    if (m == Mode::ENCRYPT)
    {
        
        permute.x = randomNumber.getRandomDouble(0 , 1);
        permute.y = randomNumber.getRandomDouble(0 , 1);
        permute.x_bar = 0;
        permute.y_bar = 0;
        permute.alpha = randomNumber.getRandomDouble(0.905 , 0.995);
        permute.beta = randomNumber.getRandomDouble(2.97 , 3.00);
        permute.myu = randomNumber.getRandomDouble(0.5 , 0.9);
        permute.r = randomNumber.getRandomDouble(1.12 , 1.18);
        permute.map = randomNumber.crngAssigner(1 , 5);
        permute.mt_seed = randomNumber.getRandomInteger(10000, 60000);
        
        if(DEBUG_PARAMETERS == 1)
        {
          cout<<"\nInitializing crng parameters for permutation\n";
          cout<<"\npermute.x = "<<permute.x;
          cout<<"\npermute.y = "<<permute.y;
          cout<<"\npermute.alpha = "<<permute.alpha;
          cout<<"\npermute.beta = "<<permute.beta;
          cout<<"\npermute.myu = "<<permute.myu;
          cout<<"\npermute.r = "<<permute.r;
          cout<<"\npermute.map = "<<int(permute.map);
          cout<<"\npermute.mt_seed = "<<permute.mt_seed<<"\n";
        }
    }
    
    if(m == Mode::DECRYPT)
    {
      if(DEBUG_PARAMETERS == 1)
      {
        cout<<"\nDecryption parameters\n";
        cout<<"\npermute.x = "<<permute.x;
        cout<<"\npermute.y = "<<permute.y;
        cout<<"\npermute.alpha = "<<permute.alpha;
        cout<<"\npermute.beta = "<<permute.beta;
        cout<<"\npermute.myu = "<<permute.myu;
        cout<<"\npermute.r = "<<permute.r;
        cout<<"\npermute.map = "<<int(permute.map);
        cout<<"\npermute.mt_seed = "<<permute.mt_seed<<"\n";
      }
    }

    //Initiliaze Parameters
    double x = permute.x;
    double y = permute.y;
    const double alpha = permute.alpha;
    const double beta = permute.beta;
    const double myu = permute.myu;
    const double r = permute.r; 
    const int mt_seed = permute.mt_seed;
    Chaos map = permute.map;
    
    CRNG crng(permute.x, permute.y, 0, 0, 0, (N - 1), permute.alpha, permute.beta, permute.myu, permute.r, permute.mt_seed, permute.map);
    
    host_vector<int> ranVec(N);
    const int exp = (int)pow(10, 9);
    int i = 0;

    auto start = steady_clock::now();
    
    for(int i = 0; i < N ; ++i)
    {
        crng.CRNGUpdateHost(x, y, 0, 0, 0, (N - 1), alpha, beta, myu, r, mt_seed, map);
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
    
    cout << "\nPERMUTATION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
     
    device_vector<int> dVec = ranVec;
    return (int *)(thrust::raw_pointer_cast(&dVec[0]));
}

/* Generate 2 vectors of N random numbers in (0,1] each */
void getDiffVecs(host_vector<double> &xVec, host_vector<double> &yVec, const int M, const int N, Diffuser &diffuse, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENCRYPT)
    {
        diffuse.x = randomNumber.getRandomDouble(0 , 0.7);
        diffuse.y = randomNumber.getRandomDouble(0 , 0.7);
        diffuse.r = randomNumber.getRandomDouble(1.12 , 1.18);
        diffuse.map = randomNumber.crngAssigner(1 , 2);
           
        if(DEBUG_PARAMETERS == 1)
        {
          cout<<"\nInitializing crng parameters for diffusion\n";
          cout<<"\ndiffuse.x = "<<diffuse.x;
          cout<<"\ndiffuse.y = "<<diffuse.y;
          cout<<"\ndiffuse.r = "<<diffuse.r;
          cout<<"\ndiffuse.map = "<<int(diffuse.map)<<"\n";
        }
    }
    
    if(m == Mode::DECRYPT)
    {
      if(DEBUG_PARAMETERS == 1)
      {
        cout<<"\nDecryption parameters\n";
        cout<<"\ndiffuse.x = "<<diffuse.x;
        cout<<"\ndiffuse.y = "<<diffuse.y;
        cout<<"\ndiffuse.r = "<<diffuse.r;
        cout<<"\ndiffuse.map = "<<int(diffuse.map)<<"\n";
      }
    }
    
    //Initiliaze Parameters
    
    double x = diffuse.x;
    double y = diffuse.y;
    const double r = diffuse.r;
    Chaos map = diffuse.map;
    
    CRNG crng(diffuse.x, diffuse.y, 0, 0, 0, 0, 0, 0, 0, diffuse.r, 0, diffuse.map);
    
    //const int exp = (int)pow(10, 9);
    int i = 0;
    auto start = steady_clock::now();
    
    for(i = 0; i < N; ++i)
    {
      crng.CRNGUpdateHost(x, y, 0, 0, 0, 0, 0, 0, 0, r, 0, map);
      xVec[i] = x;
      yVec[i] = y;
    }
    
    cout << "DIFFUSION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
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
    //cout << "Permutation: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";

    return cudaDeviceSynchronize();
}

cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m)
{
    // Initiliaze diffusion vectors
    host_vector<double> randRowX(dim[1]), randRowY(dim[1]);

    getDiffVecs(randRowX, randRowY, dim[0], dim[1], diffuse, m);

    device_vector<double> DRowX = randRowX, DRowY = randRowY;

    const double* rowXptr = (double*)(thrust::raw_pointer_cast(&DRowX[0]));
    const double* rowYptr = (double*)(thrust::raw_pointer_cast(&DRowY[0]));
    
    cout<<"\ndiffuse.r = "<<diffuse.r;
    
    //auto start = steady_clock::now();
    Wrap_Diffusion(d_img, d_imgtmp, rowXptr, rowYptr, dim, diffuse.r, int(m));
    //cout << "\nDiffusion: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";
    swap(d_img, d_imgtmp);
   
    return cudaDeviceSynchronize();
}

cudaError_t CudaImageSum(uint8_t*& d_img, uint32_t *device_sum, const int dim[])
{
  uint32_t sum = 0;
  Wrap_imageSum(d_img, device_sum, dim); 
  return cudaDeviceSynchronize();
} 

void hashPermParameters(Permuter permute, const char* hash_of_sum)
{
    /*permute.x = permute.x + (double)(hash_of_sum[0] / 1000);
    permute.y = permute.y + (double)(hash_of_sum[1] / 1000);
    permute.myu = permute.myu + (double)(hash_of_sum[2] / 1000);
    permute.alpha = permute.alpha + (double)(hash_of_sum[3] / 10000);
    permute.beta = permute.beta + (double)(hash_of_sum[4] / 10000);
    permute.r = permute.r + (double)(hash_of_sum[5] / 10000);
    permute.mt_seed = permute.mt_seed + (int)hash_of_sum[6];*/ 
    
    pVec.push_back(permute);
}

void hashDiffParameters(Diffuser diffuse, const char* hash_of_sum)
{
    diffuse.x = diffuse.x + (double)(hash_of_sum[0] / 1000);
    diffuse.y = diffuse.y + (double)(hash_of_sum[1] / 1000);
    diffuse.r = diffuse.r + (double)(hash_of_sum[5] / 10000);
    dVec.push_back(diffuse);
}

int Encrypt()
{
    kernel_WarmUp();
    Initialize();

    // Read the file and confirm it's been opened
    Mat img = imread(path.fn_img, cv::IMREAD_UNCHANGED);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
    
 
    //Resize image
    //cv::resize(img, img, cv::Size(4 , 4));
    
    
    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };
    
    //Printing image
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nOriginal image \n";
      printImageContents(img, dim[2]);
    }
    
    // Upload image to device
    uint8_t* d_img, * d_imgtmp;
    
    uint32_t host_sum;
    uint32_t *device_sum;
    size_t device_sum_size = sizeof(device_sum);
    
    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
    
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
    
    int *gpu_u;
    int *gpu_v;
    
    const char *hash_of_sum = (const char*)calloc(SHA256_DIGEST_LENGTH, sizeof(const char)); 
    
    cudaMalloc(&device_sum, device_sum_size);
    
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);
    
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);
    
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);
    
    
    // Show original image
    //imshow("Original", img);

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------ENCRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n\n";

    cudaError_t cudaStatus;

    // Encryption rounds
    for (int i = 0; i < config.rounds; i++)
    {
        cout << "X------ROUND " << i + 1 << "------X\n";
        
        /*Calculate sum of image in each round*/
        if(i > 0)
        {
          cudaStatus = CudaImageSum(d_img, device_sum, dim);
          if (cudaStatus != cudaSuccess)
          {
            cerr << "\nimageSum Failed!";
            cout<<"\nimageSum kernel error / status = "<<cudaStatus;
            return -1;
          }
          
          cudaMemcpy(&host_sum, device_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost);
          hash_of_sum = calc_sha256(host_sum);
        }
        
        // Permute Image
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
            
            if(i == 0)
            {
              pVec.push_back(permute[0]);
              pVec.push_back(permute[1]);
            }
            
            else if(i > 0)
            {
              pVecTemp.push_back(permute[0]);
              pVecTemp.push_back(permute[1]);
              hashPermParameters(permute[0], hash_of_sum);
              hashPermParameters(permute[1], hash_of_sum); 
            }
        }

        /*Diffuse image*/
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, Mode::ENCRYPT);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "\nENC_Diffusion Failed!";
            cout<<"\nENC_Diffusion kernel error / status = "<<cudaStatus;
            return -1;
        }
        
        if(i == 0)
        {
          dVec.push_back(diffuse); 
        }
            
        else if(i > 0)
        {
          dVecTemp.push_back(diffuse);
          hashDiffParameters(diffuse, hash_of_sum);
        }
    }
    
    cout<<"\nimg_sum = "<<host_sum<<"\n";
    
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
    Mat img = imread(path.fn_img_enc, cv::IMREAD_UNCHANGED);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
    
    //Resize image
    //cv::resize(img, img, cv::Size(4 , 4));
    
    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };
    
    //Printing image
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nEncrypted image \n";
      printImageContents(img, dim[2]);
    }
    
    // Upload image and LUTs to device
    uint8_t* d_img, * d_imgtmp;
    
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
    
    //uint32_t *img_sum;
    
    
    int *gpu_u;
    int *gpu_v;
    
    const char *hash_of_sum = (const char*)calloc(SHA256_DIGEST_LENGTH, sizeof(const char)); 
    
    uint32_t host_sum;
    uint32_t *device_sum;
    size_t device_sum_size = sizeof(device_sum);
    
    cudaMalloc(&device_sum, device_sum_size);
    cudaMalloc<int>(&gpu_v, lut_size_col);
    cudaMalloc<int>(&gpu_u, lut_size_row);

    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
      
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);
    
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);
    

    cout << "----------------------------------------------------------------------------------------\n";
    cout << "---------------------------------------DECRYPTION---------------------------------------\n";
    cout << "----------------------------------------------------------------------------------------\n\n";
    
    cudaError_t cudaStatus;

    // Decryption rounds
    for (int i = config.rounds - 1; i >= 0; i--)
    {
        cout << "X------ROUND " << i + 1 << "------X\n";
        
        /*Undiffuse image*/
        
        /*Calculate sum of image in each round*/
        if(i > 0)
        {
          cudaStatus = CudaImageSum(d_img, device_sum, dim);
          if (cudaStatus != cudaSuccess)
          {
            cerr << "\nimageSum Failed!";
            cout<<"\nimageSum kernel error / status = "<<cudaStatus;
            return -1;
          }
          
          cudaMemcpy(&host_sum, device_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost);
          hash_of_sum = calc_sha256(host_sum);
        }
         
        if(i == 0)
        {
          diffuse = dVec[i]; 
        }
            
        else if(i > 0)
        {
          diffuse = dVecTemp[i];
          hashDiffParameters(diffuse, hash_of_sum);
          diffuse = dVec[i];
        }
        
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, Mode::DECRYPT);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "DEC_Diffusion Failed!";
            cout<<"\nDEC_DIffusion kernel error / status = "<<cudaStatus;
            return -1;
        }
        
        //Unpermute image
        for (int j = config.rotations - 1, idx = 4 * i + 2 * j; j >= 0; j--, idx-=2)
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

    // Display decrypted image
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    imwrite(path.fn_img_dec, img);
    //imshow("Decrypted", img);
   
    cudaDeviceReset();
    waitKey(0);
    
    //Printing image
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nDecrypted image \n";
      printImageContents(img, dim[2]);
    }
    return 0;
}

#endif

