#ifndef CONFIG_H /*These two lines are to ensure no errors if the header file is included twice*/
#define CONFIG_H

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  //define something for Windows (32-bit and 64-bit, this part is common)
 std::string separator;
  #ifdef _WIN64
    //define something for Windows (64-bit only)
    separator = "\\";
  #else
    //define something for Windows (32-bit only)
    separator = "\\";
  #endif

#elif __linux__
  std::string separator = "/";
  
 
#elif __unix__ // all unices not caught above
  // Unix
   std::string separator = "/";
  
#elif defined(_POSIX_VERSION)
  // POSIX
  std::string separator = "/";

#else
  #error "Unknown compiler"
#endif

#define RESIZE_TO_DEBUG                    1
#define DEBUG_READ_WRITE                   0
#define DEBUG_VECTORS                      0

#define DEBUG_IMAGES                       1
#define DEBUG_INTERMEDIATE_IMAGES          1
#define DEBUG_MAP_PARAMETERS               0
#define DEBUG_MAP_CHOICES_ARRAY            0

#define WRITE_INPUT_THROUGH_OPENCV         1

#define ROW_COL_SWAPPING                   1
#define ROW_COL_ROTATION                   1
#define DIFFUSION                          1

#define ROUNDS_LOWER_LIMIT                 1
#define ROUNDS_UPPER_LIMIT                 2

#define X_LOWER_LIMIT                      0.10000000
#define X_UPPER_LIMIT                      0.20000000

#define Y_LOWER_LIMIT                      0.10000000
#define Y_UPPER_LIMIT                      0.20000000

#define MYU_LOWER_LIMIT                    0.50000000
#define MYU_UPPER_LIMIT                    0.90000000

#define LASM_LOWER_LIMIT                   0.40000000
#define LASM_UPPER_LIMIT                   0.90000000

#define MYU1_LOWER_LIMIT                   3.01000000
#define MYU1_UPPER_LIMIT                   3.29000000

#define MYU2_LOWER_LIMIT                   3.01000000
#define MYU2_UPPER_LIMIT                   3.30000000

#define LAMBDA1_LOWER_LIMIT                0.16000000
#define LAMBDA1_UPPER_LIMIT                0.21000000

#define LAMBDA2_LOWER_LIMIT                0.14000000
#define LAMBDA2_UPPER_LIMIT                0.15000000

#define ALPHA_LOWER_LIMIT                  0.90500000
#define ALPHA_UPPER_LIMIT                  1.00000000

#define BETA_LOWER_LIMIT                   2.97000000
#define BETA_UPPER_LIMIT                   3.20000000

#define R_LOWER_LIMIT                      1.11000000
#define R_UPPER_LIMIT                      1.19000000

#define SEED_LOWER_LIMIT                   30000
#define SEED_UPPER_LIMIT                   90000

#define MAP_LOWER_LIMIT                    1
#define MAP_UPPER_LIMIT                    5

#define LOWER_LIMIT                        0.000001
#define UPPER_LIMIT                        0.09
#define NUMBER_OF_BITS                     31
#define INIT                               100
#define BIT_RETURN(A,LOC) (( (A >> LOC ) & 0x1) ? 1:0)

namespace config
{
  /*Resized image dimensions*/
  uint32_t rows = 1024;
  uint32_t cols = 1024;
  
  /*Range of pseudorandom values to be produced by Mersenne Twister*/
  int lower_limit = 1;
  int upper_limit = (rows * cols * 3) + 1;
  
  enum class ChaoticMap
  {
    TwoDLogisticMap = 1,
    TwoDLogisticMapAdvanced,
    TwoDLogisticAdjustedSineMap,
    TwoDSineLogisticModulationMap,
    TwoDLogisticAdjustedLogisticMap
  };
  
  typedef struct
  {
    double x_init;
    double y_init;
    double myu;
  }lalm;
  
  typedef struct
  {
    double x_init;
    double y_init;
    double myu;
  }lasm;
  
  typedef struct
  {
    double x_init;
    double y_init;
    double alpha;
    double beta;
  }slmm; 
  
  typedef struct
  {
    double x_init;
    double y_init;
    double myu1;
    double myu2;
    double lambda1;
    double lambda2;
  }lma;  
   

  typedef struct
  {
    double x_init;
    double y_init;
    double r;
  }lm; 
  
  
  typedef struct 
  {
    int seed_1;
  }mt;  
  
  /*Parameter file paths*/
  std::string parameters_file = "parameters";
  std::string binary_extension = ".bin";
  std::string parameters_file_path = parameters_file + binary_extension; 
  
  /*char* constant strings for use with standard C file handling functions*/
  char *constant_parameters_file_path = const_cast<char*>(parameters_file_path.c_str());
  
}  

#endif
