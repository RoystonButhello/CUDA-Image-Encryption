#ifndef CORE_H
#define CORE_H

// Top-level encryption functions
#define DEBUG_VECTORS      0
#define DEBUG_PARAMETERS   0
#define PRINT_IMAGES       0
#define DEBUG_CONSTRUCTORS 0

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
// std::vector<Permuter> pVecTemp;
std::vector<Diffuser> dVecTemp;

/*Function Prototypes*/
static inline void printImageContents(cv::Mat image,int channels);
void Initialize();
double getParameterOffset(double value);
int* getPermVec(const int M, const int N, Permuter &permute, Mode m);

void getDiffVecs(host_vector<double> &xVec, host_vector<double> &yVec, const int M, const int N, Diffuser &diffuse, Mode m);

cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m);
cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], int iteration, int &sum_of_hash, Mode m);
cudaError_t CudaImageSum(uint8_t*& d_img, uint32_t *device_sum, const int dim[]);
static inline void getIntegerBytes(uint32_t value, char *&buffer);
static inline std::string sha256_hash_string (unsigned char hash[SHA256_DIGEST_LENGTH]);
static inline void calc_sha256(uint32_t value, int &sum_of_hash);
void hashDiffusionParameters(Diffuser &diffuse, int sum_of_hash, int i, Mode m);
int Encrypt();
int Decrypt();

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


double getParameterOffset(double value)
{
  double parameter_offset = -0.1;
  
  if(value >= 100 && value <= 255)
  {
    parameter_offset = value / 10000;
    return parameter_offset;
  } 
  
  else if(value >= 10 && value <= 99)
  {
    parameter_offset = value / 1000;
    return parameter_offset;
  }
  
  else if(value >= 9 && value <= 0)
  {
    parameter_offset = value / 100 ;
    return parameter_offset;
  }
  
  else
  {
    cout<<"\nValue out of range\n";
    printf("\nvalue = %f", value);
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
        permute.x = randomNumber.getRandomDouble(0.1 , 0.1);
        permute.y = randomNumber.getRandomDouble(0.2 , 0.2);
        permute.x_bar = 0;
        permute.y_bar = 0;
        permute.alpha = randomNumber.getRandomDouble(0.925 , 0.925);
        permute.beta = randomNumber.getRandomDouble(2.97 , 2.97);
        permute.myu = randomNumber.getRandomDouble(0.9 , 0.9);
        permute.r = randomNumber.getRandomDouble(1.16 , 1.16);
        permute.map = randomNumber.crngAssigner(2 , 2);
        permute.mt_seed = randomNumber.getRandomInteger(10000, 10000);
        
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
        diffuse.x = randomNumber.getRandomDouble(0.1 , 0.1);
        diffuse.y = randomNumber.getRandomDouble(0.2 , 0.2);
        diffuse.r = randomNumber.getRandomDouble(1.15 , 1.15);
        diffuse.map = randomNumber.crngAssigner(2 , 2);
        
        if(DEBUG_PARAMETERS == 1)
        {
          cout<<"\nINITITALIZING crng PARAMETERS FOR DIFFUSION\n";
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
        cout<<"\nDECRYPTION PARAMETERS\n";
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

cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], int iteration, int &sum_of_hash, Mode m)
{
    
    // Initiliaze diffusion vectors
    uint32_t sum_of_image = 0;
    size_t img_data_size = dim[0] * dim[1] * dim[2];
    uint8_t *img_in = (uint8_t*)calloc(img_data_size, sizeof(uint8_t));
    uint8_t *img_out = (uint8_t*)calloc(img_data_size, sizeof(uint8_t));
   
    host_vector<double> randRowX(dim[1]), randRowY(dim[1]);

    getDiffVecs(randRowX, randRowY, dim[0], dim[1], diffuse, m);

    device_vector<double> DRowX = randRowX, DRowY = randRowY;

    const double* rowXptr = (double*)(thrust::raw_pointer_cast(&DRowX[0]));
    const double* rowYptr = (double*)(thrust::raw_pointer_cast(&DRowY[0]));
    
    //cout<<"\ndiffuse.r = "<<diffuse.r;
    
    //auto start = steady_clock::now();
    Wrap_Diffusion(d_img, d_imgtmp, rowXptr, rowYptr, dim, diffuse.r, int(m));
    //cout << "\nDiffusion: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";
    
    swap(d_img, d_imgtmp);
    
    if(iteration > 0)
    {
     
     if(m == Mode::ENCRYPT)
     {
       cudaMemcpy(img_out, d_imgtmp, img_data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
     }
     
     else if(m == Mode::DECRYPT)
     {
       cudaMemcpy(img_out, d_img, img_data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost); 
     }
     
     /*cout<<"\n\n img_in = \n"; 
    
     for(int i = 0; i < img_data_size; ++i)
     {
       printf(" %d",img_in[i]);  
     }   
    
     cout<<"\n\n img_out = \n";
     for(int i = 0; i < img_data_size; ++i)
     {
       printf(" %d", img_out[i]);
     }
     cout<<"\n";*/
      
      
    for(int i = 0; i < (int)img_data_size; ++i)
    {
      sum_of_image = sum_of_image + img_out[i];
    }
      
      //calc_sha256(sum_of_image, sum_of_hash);
      cout<<"\nsum_of_image in ROUND "<<iteration + 1<<" = "<<sum_of_image<<"\n";
      //cout<<"\nsum_of_hash in ROUND "<<iteration + 1 <<" = "<<sum_of_hash<<"\n";
      
   }
   
    
    return cudaDeviceSynchronize();
}

cudaError_t CudaImageSum(uint8_t*& d_img, uint32_t *device_sum, const int dim[])
{
  uint32_t sum = 0;
  Wrap_imageSum(d_img, device_sum, dim); 
  return cudaDeviceSynchronize();
} 

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

static inline void calc_sha256(uint32_t value, int &sum_of_hash)
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
  
  for(int i = 0; i < hash_final.length(); ++i)
  {
    sum_of_hash = sum_of_hash + hash_final.at(i);
  }
  
  sum_of_hash = sum_of_hash % 256;  
  
} 

void hashDiffusionParameters(Diffuser &diffuse, int sum_of_hash, int i, Mode m)
{
    double offset = 0;
    double hash_sum = (double)sum_of_hash;
    
  if(m == Mode::ENCRYPT)
  {
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\nORIGINAL diffuse.x in ENCRYPT = %f", diffuse.x);
      printf("\nORIGINAL diffuse.y in ENCRYPT = %f", diffuse.y);
      printf("\nORIGINAL diffuse.r in ENCRYPT = %f\n", diffuse.r);
    }
      
    offset = getParameterOffset(hash_sum);
    diffuse.x = diffuse.x + offset;
    diffuse.y = diffuse.y + offset;
    diffuse.r = diffuse.r + offset;
      
    dVec.push_back(diffuse);
      
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\noffset in ENCRYPT = %f", offset);
      printf("\ndiffuse.x in ENCRYPT after PLUS = %f", diffuse.x);
      printf("\ndiffuse.y in ENCRYPT after PLUS = %f", diffuse.y);
      printf("\ndiffuse.r in ENCRYPT after PLUS = %f\n", diffuse.r);
    }
       
  }
  
  else if(m == Mode::DECRYPT)
  {
    diffuse = dVec[i];
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\nORIGINAL diffuse.x in DECRYPT  = %f", diffuse.x);
      printf("\nORIGINAL diffuse.y in DECRYPT  = %f", diffuse.y);
      printf("\nORIGINAL diffuse.r in DECRYPT  = %f\n", diffuse.r);
    }
      
      
    offset = getParameterOffset(hash_sum);
    diffuse.x = diffuse.x - offset;
    diffuse.y = diffuse.y - offset;
    diffuse.r = diffuse.r - offset;
    
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\noffset in DECRYPT = %f", offset);
      printf("\ndiffuse.x in DECRYPT after MINUS = %f", diffuse.x);
      printf("\ndiffuse.y in DECRYPT after MINUS = %f", diffuse.y);
      printf("\ndiffuse.r in DECRYPT after MINUS = %f\n", diffuse.r);
    }
      
  }
   
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
    cv::resize(img, img, cv::Size(2 , 2));
    
    
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
    
    uint32_t host_sum = 0;
    uint32_t *device_sum;
    size_t device_sum_size = sizeof(device_sum);
    
    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
    
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
    
    int *gpu_u;
    int *gpu_v;
    
    int sum_of_hash = 0; 
    
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
        
        /* Permute Image
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
        }*/
        
        
        /*Diffuse image*/
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, i, sum_of_hash, Mode::ENCRYPT);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "\nENC_Diffusion Failed!";
            cout<<"\nENC_Diffusion kernel error / status = "<<cudaStatus;
            return -1;
        }
        
        //if(i == 0)
        //{
          dVec.push_back(diffuse);
        //}
        
        //else if(i > 0)
        //{
          
          
        //}
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
    Mat img = imread(path.fn_img_enc, cv::IMREAD_UNCHANGED);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
    
    //Resize image
    cv::resize(img, img, cv::Size(2 , 2));
    
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
    
    int sum_of_hash = 0;
    
    uint32_t host_sum = 0;
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
        //if(i == 0)
        //{
          diffuse = dVec[i];
        //}
        
        //else if(i > 0)
        //{
          
        //}
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, i, sum_of_hash, Mode::DECRYPT);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "DEC_Diffusion Failed!";
            cout<<"\nDEC_DIffusion kernel error / status = "<<cudaStatus;
            return -1;
        }
        
        
        /*Unpermute image
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
        }*/
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
    
    cout<<"\n\n\ndVec :";
    for(Diffuser diffuse: dVec)
    {
      printf("\ndiffuse.x = %f", diffuse.x);
      printf("\ndiffuse.y = %f", diffuse.y);
      printf("\ndiffuse.r = %f", diffuse.r);
    }
    
    return 0;
}

#endif


