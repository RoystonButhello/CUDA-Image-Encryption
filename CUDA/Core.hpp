#ifndef CORE_H
#define CORE_H

// Top-level encryption functions
#define DEBUG_VECTORS                    0
#define DEBUG_PARAMETERS                 0
#define DEBUG_KEY                        0
#define DEBUG_PARAM_MOD                  0
#define PRINT_IMAGES                     0
#define DEBUG_CONSTRUCTORS               0

#define X_LOWER_LIMIT                    0.1
#define X_UPPER_LIMIT                    0.7
#define Y_LOWER_LIMIT                    0.1
#define Y_UPPER_LIMIT                    0.7
#define ALPHA_LOWER_LIMIT                0.905
#define ALPHA_UPPER_LIMIT                0.985
#define BETA_LOWER_LIMIT                 2.97
#define BETA_UPPER_LIMIT                 3.00
#define MYU_LOWER_LIMIT                  0.50
#define MYU_UPPER_LIMIT                  0.80
#define R_LOWER_LIMIT                    1.15
#define R_UPPER_LIMIT                    1.17
#define MAP_LOWER_LIMIT                  1
#define MAP_UPPER_LIMIT                  5
#define PERMUTE_PROPAGATION_LOWER_LIMIT  4000
#define PERMUTE_PROPAGATION_UPPER_LIMIT  5000
#define DIFFUSE_PROPAGATION_LOWER_LIMIT  8000
#define DIFFUSE_PROPAGATION_UPPER_LIMIT  10000  

#define PERM_OFFSET_LOWER_LIMIT          0.00001  
#define PERM_OFFSET_UPPER_LIMIT          0.00002
#define DIFF_OFFSET_LOWER_LIMIT          0.00001
#define DIFF_OFFSET_UPPER_LIMIT          0.00002 

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
uint32_t host_hash_sum_plain = 0;
uint32_t hash_sum_byte_plain = 0;

/*Function Prototypes*/
static inline void printImageContents(cv::Mat image,int channels);
void Initialize(std::string file, int rounds, int rotations_per_round);
double getParameterOffset(double value);
int* getPermVec(const int M, const int N, Permuter &permute, Mode m);

void getDiffVecs(host_vector<double> &xVec, host_vector<double> &yVec, const int M, const int N, Diffuser &diffuse, Mode m);

void modifyPermutationParameters(Permuter &permute, double permutation_parameter_modifier, double perm_param_offset);
void modifyDiffusionParameters(Diffuser &diffuse, double diffusion_parameter_modifier, double diff_param_offset);

cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m);
cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], uint32_t diffuse_propagation_factor, Mode m);
cudaError_t CudaImageSumReduce(uint8_t *img, uint32_t *device_result, uint32_t &host_sum, const int dim[]);

static inline void getIntegerBytes(uint32_t value, unsigned char *&buffer);
static inline std::string sha256_hash_string (unsigned char hash[SHA256_DIGEST_LENGTH]);
static inline void calc_sum_of_hash(uint32_t value, uint32_t &hash_sum_byte);
void reverseChangePropagation(std::vector<Permuter> &pVec, std::vector<Diffuser> &dVec, uint32_t hash_sum_byte, Mode m);
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
    return parameter_offset;
  }
  
  return parameter_offset;
}


/**
 * Generates vector of N random numbers in [0,M]. Takes number of rows and columns, object of Permute class and Mode of operation as parameters 
 */
int* getPermVec(const int M, const int N, Permuter &permute, Mode m)
{
    
    //Initiliaze CRNG
    if (m == Mode::ENCRYPT)
    {
        auto start_permgen = steady_clock::now();
        permute.x = randomNumber.getRandomDouble(X_LOWER_LIMIT , X_UPPER_LIMIT);
        permute.y = randomNumber.getRandomDouble(Y_LOWER_LIMIT , Y_UPPER_LIMIT);
        permute.x_bar = randomNumber.getRandomDouble(X_LOWER_LIMIT, X_UPPER_LIMIT);
        permute.y_bar = randomNumber.getRandomDouble(Y_LOWER_LIMIT, Y_UPPER_LIMIT);
        permute.alpha = randomNumber.getRandomDouble(ALPHA_LOWER_LIMIT , ALPHA_UPPER_LIMIT);
        permute.beta = randomNumber.getRandomDouble(BETA_LOWER_LIMIT , BETA_UPPER_LIMIT);
        permute.myu = randomNumber.getRandomDouble(MYU_LOWER_LIMIT , MYU_UPPER_LIMIT);
        permute.r = randomNumber.getRandomDouble(R_LOWER_LIMIT , R_UPPER_LIMIT);
        permute.map = randomNumber.crngAssigner(MAP_LOWER_LIMIT , MAP_UPPER_LIMIT);
        offset.permute_param_offset = randomNumber.getRandomDouble(PERM_OFFSET_LOWER_LIMIT, PERM_OFFSET_UPPER_LIMIT);
        cout << "Generate permutation params & offset: " << (duration_cast<milliseconds>(steady_clock::now() - start_permgen).count()) << "ms\n";
	auto start_pmod = steady_clock::now(); 
        modifyPermutationParameters(permute, offset.permute_param_modifier, offset.permute_param_offset);
        cout << "Modify permutation parameters:" << (duration_cast<nanoseconds>(steady_clock::now() - start_pmod).count()) << "ns\n";
        if(DEBUG_PARAMETERS == 1)
        {
          printf("\npermute.x = %f", permute.x);
          printf("\npermute.y = %f", permute.y);
          printf("\npermute.r = %f", permute.r);
          printf("\npermute.x_bar = %f", permute.x_bar);
          printf("\npermute.y_bar = %f", permute.y_bar);
          printf("\npermute.alpha = %f", permute.alpha);
          printf("\npermute.beta = %f", permute.beta);
          printf("\npermute.map = %d", int(permute.map));
          
        }
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

    auto start = steady_clock::now();
    
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
    
    cout << "\nPERMUTATION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
     
    device_vector<int> dVec = ranVec;
    return (int *)(thrust::raw_pointer_cast(&dVec[0]));
}

/**
 * Generate 2 vectors of N random numbers in (0,1] range. Takes both diffusion vectors, number of rows and columns, Diffuse class object and mode of operation as parameters 
 */
void getDiffVecs(host_vector<double> &xVec, host_vector<double> &yVec, const int M, const int N, Diffuser &diffuse, Mode m)
{
    //Initiliaze CRNG
    if (m == Mode::ENCRYPT)
    {
        auto start_diffgen = steady_clock::now();
        diffuse.x = randomNumber.getRandomDouble(X_LOWER_LIMIT , X_UPPER_LIMIT);
        diffuse.y = randomNumber.getRandomDouble(Y_LOWER_LIMIT , Y_UPPER_LIMIT);
        diffuse.x_bar = randomNumber.getRandomDouble(X_LOWER_LIMIT, X_UPPER_LIMIT);
        diffuse.y_bar = randomNumber.getRandomDouble(Y_LOWER_LIMIT, Y_UPPER_LIMIT);
        diffuse.alpha = randomNumber.getRandomDouble(ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT);
        diffuse.beta = randomNumber.getRandomDouble(BETA_LOWER_LIMIT, BETA_UPPER_LIMIT);
        diffuse.myu = randomNumber.getRandomDouble(MYU_LOWER_LIMIT, MYU_UPPER_LIMIT);
        diffuse.r = randomNumber.getRandomDouble(R_LOWER_LIMIT , R_UPPER_LIMIT);
        diffuse.map = randomNumber.crngAssigner(1 , 5);
        offset.diffuse_param_offset = randomNumber.getRandomDouble(DIFF_OFFSET_LOWER_LIMIT, DIFF_OFFSET_UPPER_LIMIT);
        cout << "Generate diffusion parameters and offset: " << (duration_cast<milliseconds>(steady_clock::now() - start_diffgen).count()) << "ms\n";

        auto start_dmod = steady_clock::now();
        modifyDiffusionParameters(diffuse, offset.diffuse_param_modifier, offset.diffuse_param_offset);
        cout << "Modify diffusion parameters: " << (duration_cast<nanoseconds>(steady_clock::now() - start_dmod).count()) << "ns\n";

        if(DEBUG_PARAMETERS == 1)
        {
          printf("\ndiffuse.x = %f", diffuse.x);
          printf("\ndiffuse.y = %f", diffuse.y);
          printf("\ndiffuse.r = %f", diffuse.r);
          printf("\ndiffuse.x_bar = %f", diffuse.x_bar);
          printf("\ndiffuse.y_bar = %f", diffuse.y_bar);
          printf("\ndiffuse.alpha = %f", diffuse.alpha);
          printf("\ndiffuse.beta = %f", diffuse.beta);
          printf("\ndiffuse.map = %d", int(diffuse.map));
          
        }
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
    auto start = steady_clock::now();
    
    for(i = 0; i < N; ++i)
    {
      crng.CRNGUpdateHost(x, y, 0, 0, alpha, beta, myu, r, map);
      xVec[i] = x;
      yVec[i] = y;
    }
    
    cout << "DIFFUSION CRNG: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n";
}

/**
 * Modifies all permutation parameters with Permute Parameter Modifier and Permute Parameter Offset in each permutation round. Takes an object of permute class, the permute parmeter modifier and the permute parameter offset as arguments
 */   
void modifyPermutationParameters(Permuter &permute, double permutation_parameter_modifier, double perm_param_offset)
{
  switch(int(permute.map))
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
      permute.x_bar = permute.x_bar + permutation_parameter_modifier + perm_param_offset;
      permute.y_bar = permute.y_bar + permutation_parameter_modifier + perm_param_offset;
      permute.myu = permute.myu + permutation_parameter_modifier + perm_param_offset;  
    }
    break;
    
    default:cout<<"\nInvalid map choice for permutation\n"; 
  }
}

/**
 * Modifies all diffusion parameters with Diffuse Parameter Modifier and Diffuse Parameter Offset in each encryption round. Takes an object of Diffuse class, the diffusion parameter modifier and the diffuse parameter offset as arguments
 */
void modifyDiffusionParameters(Diffuser &diffuse, double diffusion_parameter_modifier, double diff_param_offset)
{
  switch(int(diffuse.map))
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
      diffuse.x_bar = diffuse.x_bar + diffusion_parameter_modifier + diff_param_offset;
      diffuse.y_bar = diffuse.y_bar + diffusion_parameter_modifier + diff_param_offset;
      diffuse.myu = diffuse.myu + diffusion_parameter_modifier + diff_param_offset;  
    }
    break;
    
    default:cout<<"\nInvalid map choice for permutation\n"; 
  }
}


/**
 * Top level function for permutation that calls getPermVec() to generate permutation vectors and Wrap_RotatePerm() to run the permutation CUDA kernel. Takes a 1D N X M input image, image dimensions and mode of operation as arguments
 */
cudaError_t CudaPermute(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], Mode m)
{
    // Generate permutation vectors
    auto ptrU = getPermVec(dim[0], dim[1], permute[0], m);
    auto ptrV = getPermVec(dim[1], dim[0], permute[1], m);
    
    // Set grid and block data_size
    const dim3 grid(dim[0], dim[1], 1);
    const dim3 block(dim[2], 1, 1);

    //auto start = steady_clock::now();
    //Calling CUDA permutation kernel
    Wrap_RotatePerm(d_img, d_imgtmp, ptrU, ptrV, grid, block,int(m));
    //Transferring the output image vector into input image vector 
    swap(d_img, d_imgtmp);
    //cout << "Permutation: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";

    return cudaDeviceSynchronize();
}

/**
 * Top level function for diffusion and self - xor that calls getDiffVecs() to generate diffusion vectors and Wrap_Diffusion() to run the diffusion CUDA kernel and the self - xor CUDA kernel. Takes a 1D N X M input image, image dimensions, diffuse propagation factor and mode of operation as arguments
 */
cudaError_t CudaDiffuse(uint8_t*& d_img, uint8_t*& d_imgtmp, const int dim[], uint32_t diffuse_propagation_factor, Mode m)
{
    // Initiliaze diffusion vectors    
    host_vector<double> randRowX(dim[1]), randRowY(dim[1]);

    getDiffVecs(randRowX, randRowY, dim[0], dim[1], diffuse, m);

    device_vector<double> DRowX = randRowX, DRowY = randRowY;

    const double* rowXptr = (double*)(thrust::raw_pointer_cast(&DRowX[0]));
    const double* rowYptr = (double*)(thrust::raw_pointer_cast(&DRowY[0]));
        
    //auto start = steady_clock::now();
    //Calling CUDA diffusion kernel and self - xor kernel 
    Wrap_Diffusion(d_img, d_imgtmp, rowXptr, rowYptr, dim, diffuse.r, int(m), diffuse_propagation_factor);
    //Transferring the output image vector into input image vector
    swap(d_img, d_imgtmp);
    //cout << "\nDiffusion: " << (duration_cast<microseconds>(steady_clock::now() - start).count()) << "us\n\n";
    
    return cudaDeviceSynchronize();
}


/**
 * Top level function for calculating sum of image. Calls Wrap_imageSumReduce() CUDA kernel. Takes a 1D N X M input image, image sum and image dimensions as arguments 
 */
cudaError_t CudaImageSumReduce(uint8_t *img, uint32_t *device_result, uint32_t &host_sum, const int dim[])
{
  Wrap_imageSumReduce(img, device_result, dim);
  cudaMemcpy(&host_sum, device_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  return cudaDeviceSynchronize(); 
}

/**
 * Gets each byte of each element of a vector. Takes the size of the vector's data type and the vector as arguments
 */
static inline void getIntegerBytes(uint32_t value, unsigned char *&buffer)
{
   for(int i = 0; i < sizeof(value); ++i)
   {
     buffer[i] = (value >> (8 * i)) & 0xff;
   }
} 

/**
 * Converts an 8-bit unsigned char SHA256 array into a SHA256 std::string. Takes the 8-bit unsigned char hash array of length 64 as an argument
 */
static inline std::string sha256_hash_string (unsigned char hash[SHA256_DIGEST_LENGTH])
{
    
  stringstream ss;
  for(int i = 0; i < SHA256_DIGEST_LENGTH; i++)
  {
      ss << hex << setw(2) << setfill('0') << (int)hash[i];
  }
    
  return ss.str();
}

/**
 * Calculates the SHA256 hash of a 32-bit unsigned integer, takes the sum of all the bytes of the hash, divides that sum by 256 and takes its remainder. Takes the value for wwhich the hash has to be computed and another integer where the result will be stored as arguments
 */
static inline void calc_sum_of_hash(uint32_t value, uint32_t &hash_sum_byte)
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
  
  hash_sum_byte = (uint8_t)x;
} 

/**
 * Modifies or reverses modification of all permutation and diffusion parameters by using the Reverse Change Propagation offset. Takes the permute parameter vector, the diffuse parameter vector, the Reverse Change Propagation offset and the mode of operation as arguments
 */
void reverseChangePropagation(std::vector<Permuter> &pVec, std::vector<Diffuser> &dVec, uint32_t hash_sum_byte, Mode m)
{
  double reverse_change_propagation_offset = 0;
  Permuter permute;
  Diffuser diffuse;
  
  if(m == Mode::ENCRYPT)
  {  
    //Getting reverse change propagation offset from output cipher image
    reverse_change_propagation_offset = getParameterOffset(hash_sum_byte);
    
    //Modifying all diffusion parameters
    for(int i = 0; i < dVec.size(); ++i)
    {
      diffuse = dVec[i];
      
      diffuse.x = diffuse.x + reverse_change_propagation_offset;
      diffuse.y = diffuse.y + reverse_change_propagation_offset;
      diffuse.alpha = diffuse.alpha + reverse_change_propagation_offset;
      diffuse.beta = diffuse.beta + reverse_change_propagation_offset;
      diffuse.myu = diffuse.myu + reverse_change_propagation_offset;
      diffuse.r = diffuse.r + reverse_change_propagation_offset;
      
      dVec[i] = diffuse;
    }   
    
    //Modifying all permutation parameters
    for(int i = 0; i < pVec.size(); ++i)
    {
      permute = pVec[i];
      
      permute.x = permute.x + reverse_change_propagation_offset;
      permute.y = permute.y + reverse_change_propagation_offset;
      permute.r = permute.r + reverse_change_propagation_offset;
      permute.alpha = permute.alpha + reverse_change_propagation_offset;
      permute.beta = permute.beta + reverse_change_propagation_offset;
      permute.myu = permute.myu + reverse_change_propagation_offset;
      
      pVec[i] = permute;
    }
  }
  
  else if(m == Mode::DECRYPT)
  {
    //Getting reverse change propagation offset from input cipher image
    reverse_change_propagation_offset = getParameterOffset(hash_sum_byte);
    
    //Unmodifying all diffusion parameters
    for(int i = 0; i < dVec.size(); ++i)
    {
      diffuse = dVec[i];

      
      diffuse.x = diffuse.x - reverse_change_propagation_offset;
      diffuse.y = diffuse.y - reverse_change_propagation_offset;
      diffuse.alpha = diffuse.alpha - reverse_change_propagation_offset;
      diffuse.beta = diffuse.beta - reverse_change_propagation_offset;
      diffuse.myu = diffuse.myu - reverse_change_propagation_offset;
      diffuse.r = diffuse.r - reverse_change_propagation_offset;
      
      dVec[i] = diffuse;  
    }
    
    //Unodifying all permutation parameters
    for(int i = 0; i < pVec.size(); ++i)
    {
      permute = pVec[i];
      
      permute.x = permute.x - reverse_change_propagation_offset;
      permute.y = permute.y - reverse_change_propagation_offset;
      permute.r = permute.r - reverse_change_propagation_offset;
      permute.alpha = permute.alpha - reverse_change_propagation_offset;
      permute.beta = permute.beta - reverse_change_propagation_offset;
      permute.myu = permute.myu - reverse_change_propagation_offset;
      
      pVec[i] = permute;
    } 
  }  
}

/**
 * Prints all the elements of an unsigned 8-bit integer array. Takes the array and its length as arguments.
 */
void printInt8Array(uint8_t *array, int length)
{
  for(int i = 0; i < length; ++i)
  {
    printf(" %d", array[i]);
  }
}

/**
 * Returns the power of 10 of a 32-bit unsigned integer value. Takes the said value as an argument
 */
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

/**
 * Calculates the size of the encryption key. Takes the permute parameter vector and the diffuse parameter vector as arguments 
 */
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

/**
 * Encrypts images. Takes the absolute file path of the image, the number of encryption rounds and the number of permutation rounds per encryption round as arguments 
 */
int Encrypt(std::string file, int rounds, int rotations)
{
    kernel_WarmUp();
    Initialize(file, rounds, rotations);

    // Read the file and confirm it's been opened
    auto start_read = steady_clock::now();
    Mat img = imread(path.fn_img, cv::IMREAD_UNCHANGED);
    cout << "Read plain image: " << (duration_cast<milliseconds>(steady_clock::now() - start_read).count()) << "ms\n";
    
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

    uint32_t host_hash_sum_ENC = 0;
    uint32_t *device_hash_sum_ENC, *device_hash_sum_plain;
    
    size_t device_hash_sum_size = sizeof(device_hash_sum_ENC);
    
    size_t data_size = img.rows * img.cols * img.channels() * sizeof(uint8_t);
    
    size_t lut_size_row = dim[1] * sizeof(int);
    size_t lut_size_col = dim[0] * sizeof(int);
    
    int *gpu_u;
    int *gpu_v;
    
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
    cout << "Calculating sum of plain image: " << (duration_cast<microseconds>(steady_clock::now() - start_sumplain).count()) << "us\n";
    
    
    //Calculate sum of the sha256 hash of the sum of the plain image
    auto start_plain_hashcalc = steady_clock::now();
    calc_sum_of_hash(host_hash_sum_plain, hash_sum_byte_plain);
    cout << "Calculate hash sum of plain image: " << (duration_cast<microseconds>(steady_clock::now() - start_plain_hashcalc).count()) << "us\n";
    
    auto start_propagationcalc = steady_clock::now();
    //Factor to induce propagation in permutation vector generation parameters
    propagator.permute_propagation_factor = hash_sum_byte_plain ^ randomNumber.getRandomUnsignedInteger32(PERMUTE_PROPAGATION_LOWER_LIMIT, PERMUTE_PROPAGATION_UPPER_LIMIT);
    //Factor to induce forward propagation in diffusion vector generation parameters and diffusion kernel
    propagator.diffuse_propagation_factor = hash_sum_byte_plain ^ randomNumber.getRandomUnsignedInteger32(DIFFUSE_PROPAGATION_LOWER_LIMIT, DIFFUSE_PROPAGATION_UPPER_LIMIT);
    cout << "Calculate permute and diffuse propagation factors: " << (duration_cast<microseconds>(steady_clock::now() - start_propagationcalc).count()) << "us\n";
    
    //Permutation and diffusion parameter modifiers 
    offset.permute_param_modifier = getParameterOffset(propagator.permute_propagation_factor);
    offset.diffuse_param_modifier = getParameterOffset(propagator.diffuse_propagation_factor);
    
    if(cudaStatus != cudaSuccess)
    {
      cerr << "\nimage sum Failed!";
      cout<<"\nimage sum kernel error / status = "<<cudaStatus;
      return -1;
    }
    
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\nhost_sum_plain = %d", host_hash_sum_plain);
      printf("\nhash_sum_byte_plain = %d", hash_sum_byte_plain);
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
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, propagator.diffuse_propagation_factor, Mode::ENCRYPT);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "\nENC_Diffusion Failed!";
            cout<<"\nENC_Diffusion kernel error / status = "<<cudaStatus;
            return -1;
        }
       
        dVec.push_back(diffuse);
  }
    
    cout << "DURATION OF ENCRYPTION: " << (duration_cast<milliseconds>(steady_clock::now() - start_enc).count()) << "ms\n";
    
    //Copy encrypted image from device memory to host memory 
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    
    //Write encrypted image to disk
    auto start_write = steady_clock::now();
    imwrite(path.fn_img_enc, img);
    cout << "Write encrypted image: " << (duration_cast<milliseconds>(steady_clock::now() - start_write).count()) << "ms\n";

    auto start_imagesum = steady_clock::now();    
    //Calculate sum of encrypted image
    cudaStatus = CudaImageSumReduce(d_img, device_hash_sum_ENC, host_hash_sum_ENC, dim);
    cout << "Calculate sum of encrypted image: " << (duration_cast<microseconds>(steady_clock::now() - start_imagesum).count()) << "us\n";
  
    if (cudaStatus != cudaSuccess)
    {
      cerr << "\nimage sum Failed!";
      cout<<"\nimage sum kernel error / status = "<<cudaStatus;
      return -1;
    }

    auto start_hash = steady_clock::now(); 
    //Calculate sum of the sha256 hash of the sum of the encrypted image
    calc_sum_of_hash(host_hash_sum_ENC, hash_sum_byte_ENC);
    cout << "Calculate sum of hash of encrypted image: " << (duration_cast<microseconds>(steady_clock::now() - start_hash).count()) << "us\n";
   
    auto start_modification = steady_clock::now(); 
    //Modify the parameters by adding Reverse Change Propagation Offset to them
    reverseChangePropagation(pVec, dVec, hash_sum_byte_ENC, Mode::ENCRYPT);
    cout << "Modify all parameters for reverse change propagation: " << (duration_cast<microseconds>(steady_clock::now() - start_modification).count()) << "us\n";
    
    if(DEBUG_PARAMETERS == 1)
    {    
      printf("\nhost_hash_sum_ENC = %d", host_hash_sum_ENC);
      printf("\nhash_sum_byte_ENC = %d\n", hash_sum_byte_ENC);
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

/**
 * Performs image decryption. Takes no arguments
 */
int Decrypt()
{
    
    kernel_WarmUp();

    // Read the file and confirm it's been opened
    auto start_read = steady_clock::now(); 
    cv::Mat img = cv::imread(path.fn_img_enc, cv::IMREAD_UNCHANGED);
    cout << "Read the encrypted image: " << (duration_cast<milliseconds>(steady_clock::now() - start_read).count()) << "ms\n";
    
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
    
    uint32_t hash_sum_byte_DEC = 0;
    
    uint32_t host_hash_sum_DEC = 0;
    uint32_t *device_hash_sum_DEC;
   
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
      cout<<"\nimage sum kernel error / status = "<<cudaStatus;
      return -1;
    }
    
    auto start_hash = steady_clock::now();
    //Calculate sum of hash of sum of encrypted image
    calc_sum_of_hash(host_hash_sum_DEC, hash_sum_byte_DEC);
    cout << "Compute SHA256 hash sum of encrypted image " << (duration_cast<microseconds>(steady_clock::now() - start_hash).count()) << "us\n";
    
    auto start_recovery = steady_clock::now();
    //Recover all permutation and diffusion parameters by subtracting from said parameters, the Reverse Propagation Offset
    reverseChangePropagation(pVec, dVec, hash_sum_byte_DEC, Mode::DECRYPT);
    cout << "Recover encryption parameters: " << (duration_cast<microseconds>(steady_clock::now() - start_recovery).count()) << "us\n";
    
    if(DEBUG_PARAMETERS == 1)
    {
      printf("\nhost_hash_sum_DEC = %d", host_hash_sum_DEC);
      printf("\nhash_sum_byte_DEC = %d\n", hash_sum_byte_DEC);
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
        cudaStatus = CudaDiffuse(d_img, d_imgtmp, dim, propagator.diffuse_propagation_factor, Mode::DECRYPT);
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

    cout << "DURARTION OF DECRYPTION: " << (duration_cast<milliseconds>(steady_clock::now() - start_dec).count()) << "ms\n";
    
    //Copy decrypted image from device memory to host memory
    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    
    //Write decrypted image to disk
    auto start_write = steady_clock::now();
    imwrite(path.fn_img_dec, img);
    cout << "Write Decrypted Image: " << (duration_cast<milliseconds>(steady_clock::now() - start_write).count()) << "ms\n";
    
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

