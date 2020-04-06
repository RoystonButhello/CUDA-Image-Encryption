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

#define RESIZE_TO_DEBUG            1
#define DEBUG_VECTORS              0
#define DEBUG_IMAGES               1
#define DEBUG_INTERMEDIATE_IMAGES  0
#define PRINT_TIMING               0
#define PRINT_IMAGES               0
#define LOWER_LIMIT                0.000001
#define UPPER_LIMIT                0.09
#define NUMBER_OF_BITS             16
#define INIT                       100
#define BIT_RETURN(A,LOC) (( (A >> LOC ) & 0x1) ? 1:0)

namespace config
{
  
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
    double myu1;
    double myu2;
    double lambda1;
    double lambda2;
    double randnum;
  }lma;  
   
  typedef struct
  {
    double x_init;
    double y_init;
    double alpha;
    double beta;
  }slmm; 
 
  slmm slmm_map;
  lma lma_map;
  lasm lasm_map;
  uint32_t rows = 1024;
  uint32_t cols = 1024;
  int lower_limit = 1;
  int upper_limit = (rows * cols * 3) + 1;
  int seed_lut_gen = 1234567890;
  int seed_row_rotate = 3712908712;
  int seed_col_rotate = 380219812;
  
  
  std::string image_name = "airplane";
  std::string encrypted_image = image_name + "_encrypted_";
  std::string decrypted_image = image_name + "_decrypted_";
  std::string row_col_permuted_image = image_name + "_row_col_permuted_";
  std::string row_col_unpermuted_image = image_name + "_row_col_unpermuted_";
  std::string extension = ".png";
  std::string input = "input";
  std::string output = "output"; 
  
  std::string input_image_path = input + separator + image_name + extension;
  std::string encrypted_image_path = output + separator + encrypted_image + std::to_string(rows) + "_" + std::to_string(cols) + extension;
  std::string decrypted_image_path = output + separator + decrypted_image + std::to_string(rows) + "_" + std::to_string(cols) + extension;   
}  

#endif