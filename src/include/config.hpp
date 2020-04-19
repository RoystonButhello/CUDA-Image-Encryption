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

#define RESIZE_TO_DEBUG              1
#define DEBUG_VECTORS                0

#define DEBUG_IMAGES                 1
#define DEBUG_INTERMEDIATE_IMAGES    1

#define PRINT_TIMING                 1
#define PRINT_IMAGES                 0

#define ROW_COL_SWAP                 1
#define ROW_COL_ROTATE               1
#define DIFFUSION                    1

#define TWOD_LALM                    1
#define TWOD_LASM                    1
#define TWOD_SLMM                    1
#define TWOD_LMA                     1
#define TWOD_LM                      1

#define LOWER_LIMIT                  0.000001
#define UPPER_LIMIT                  0.09

#define ROUNDS_LOWER_LIMIT           1
#define ROUNDS_UPPER_LIMIT           2

#define ROTATE_ROUNDS_LOWER_LIMIT    1
#define ROTATE_ROUNDS_UPPER_LIMIT    2

#define SWAP_ROUNDS_LOWER_LIMIT      1
#define SWAP_ROUNDS_UPPER_LIMIT      2

#define DIFFUSION_ROUNDS_LOWER_LIMIT 1
#define DIFFUSION_ROUNDS_UPPER_LIMIT 2

#define SEED_LOWER_LIMIT             3000000
#define SEED_UPPER_LIMIT             1073741824       

#define X_LOWER_LIMIT                0.10000000
#define X_UPPER_LIMIT                0.90000000

#define Y_LOWER_LIMIT                0.10000000
#define Y_UPPER_LIMIT                0.90000000

#define MYU_LOWER_LIMIT              0.50000000
#define MYU_UPPER_LIMIT              0.90000000

#define LASM_LOWER_LIMIT             0.40000000
#define LASM_UPPER_LIMIT             0.90000000

#define MYU1_LOWER_LIMIT             3.01000000
#define MYU1_UPPER_LIMIT             3.29000000

#define MYU2_LOWER_LIMIT             3.01000000
#define MYU2_UPPER_LIMIT             3.30000000

#define LAMBDA1_LOWER_LIMIT          0.16000000
#define LAMBDA1_UPPER_LIMIT          0.21000000

#define LAMBDA2_LOWER_LIMIT          0.14000000
#define LAMBDA2_UPPER_LIMIT          0.15000000

#define ALPHA_LOWER_LIMIT            0.90500000
#define ALPHA_UPPER_LIMIT            1.00000000

#define BETA_LOWER_LIMIT             2.97000000
#define BETA_UPPER_LIMIT             3.20000000

#define R_LOWER_LIMIT                1.11000000
#define R_UPPER_LIMIT                1.19000000

              
#define NUMBER_OF_BITS               16
#define INIT                         100
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
 
  typedef struct 
  {
    
    slmm slmm_parameters;
    lasm lasm_parameters;
    lma  lma_parameters;
    uint8_t rotation_rounds;
    uint8_t swap_rounds;
    uint8_t diffusion_rounds;
    uint32_t seed_lut_gen_1;
    uint32_t seed_lut_gen_2;
    uint32_t seed_row_rotate;
    uint32_t seed_col_rotate; 
  }algorithm;  
  
  slmm slmm_map;
  
  uint32_t rows = 1024;
  uint32_t cols = 1024;
  
  int lower_limit = 1;
  int upper_limit = (rows * cols * 3) + 1;
  int seed_lut_gen = 1234567890;
  int seed_row_rotate = 3712908712;
  int seed_col_rotate = 380219812;  

  enum class ChaoticMap
  {
    TwoDLogisticMap,
    TwoDLogisticMapAdvanced,
    TwoDLogisticAdjustedSineMap,
    TwoDSineLogisticModulationMap,
    TwoDLogisticAdjustedLogisticMap
  };
  
  ChaoticMap chaotic_map = ChaoticMap::TwoDSineLogisticModulationMap;
  
  clock_t initialize_image_paths_start = std::clock();
  std::string image_name = "airplane";
  std::string row_col_rotated_image = image_name + "_row_col_rotated_";
  std::string row_col_unrotated_image = image_name + "_row_col_unrotated_";
  std::string row_col_swapped_image = image_name + "_row_col_swapped_";
  std::string row_col_unswapped_image = image_name + "_row_col_unswapped_";
  std::string diffused_image = image_name + "_diffused_";
  std::string undiffused_image = image_name + "_undiffused_";
  std::string string_rows = std::to_string(rows);
  std::string string_cols = std::to_string(cols);
  std::string final_encrypted = "_final_encrypted";
  std::string final_decrypted = "_final_decrypted";
    
  std::string extension = ".png";
  std::string input = "input";
  std::string output = "output"; 
  
  std::string input_image_path = input + separator + image_name + extension;
  std::string row_col_rotated_image_path = output + separator + row_col_rotated_image + string_rows + "_" + string_cols + extension;
  std::string row_col_unrotated_image_path = output + separator + row_col_unrotated_image + string_rows + "_" + string_cols + final_decrypted + extension;
  std::string row_col_swapped_image_path = output + separator + row_col_swapped_image + string_rows + "_" + string_cols + extension;   
  
  std::string row_col_unswapped_image_path = output + separator + row_col_unswapped_image + string_rows + "_" + string_cols + extension;
  
  std::string diffused_image_path = output + separator + diffused_image +string_rows + "_" + string_cols + final_encrypted + extension;
  std::string undiffused_image_path = output + separator + undiffused_image + string_rows + "_" + string_cols + extension;
  clock_t initialize_image_paths_end = std::clock();
  long double img_path_init_time = 1000.0 * (initialize_image_paths_end - initialize_image_paths_start) / CLOCKS_PER_SEC;
    
}  

#endif
