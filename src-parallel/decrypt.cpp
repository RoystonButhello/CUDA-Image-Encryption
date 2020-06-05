#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"
#include "include/io.hpp"

int main(int argc,char *argv[])
{
  /**
   * Get encrypted file path
   */
  std::string input_image_path = std::string(argv[1]);
  
  if(input_image_path == "")
  {
    cout<<"\nNo image name specified.Please specify an image name.\nExiting...";
    exit(0);
  }
  
  /**
   * Get encrypted image name from file path
   */
  std::string image_name = common::getFileNameFromPath(input_image_path);
  
  input_image_path = "";
  input_image_path = image_name + "_diffused" + ".png";
    
  auto start = std::chrono::system_clock::now();
    
  cv::Mat image;
  image = cv::imread(input_image_path,cv::IMREAD_UNCHANGED);
   
  if(!image.data)
  {
    cout<<"\nCould not open image from "<<input_image_path<<" \nExiting...";
    exit(0);
  }

  
  uint32_t m = 0,n = 0,channels = 0, total = 0;
  double alpha = 0.00,beta = 0.00;
  int number_of_rotation_rounds = 0, number_of_swapping_rounds = 0,i = 0;
  long ptr_position = 0; 
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  int fread_status = 9,fseek_status = 9,seed = 0;  

  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  uint32_t *map_choice_array = (uint32_t*)calloc(6,sizeof(uint32_t)); 
  uint32_t *map_array = (uint32_t*)calloc(15,sizeof(uint32_t));
  
  /** 
   * Reading number of rotation rounds
   */
  FILE *infile = fopen(config::constant_parameters_file_path,"rb");
  
  if(infile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for reading\nExiting...";
    exit(0);
  }
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\npointer position before reading the number of rotation rounds = "<<ptr_position;
  }
  
  fread_status = fread(&number_of_rotation_rounds,sizeof(number_of_rotation_rounds),1,infile);
  ptr_position =  ftell(infile);
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\npointer position after reading the number of rotation rounds = "<<ptr_position; 
  }
  
  fclose(infile);
  
  /**
   * Reading number of swapping rounds
   */
  infile = fopen(config::constant_parameters_file_path,"rb");
  
  if(infile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for reading\nExiting...";
    exit(0);
  }
  
  if(ptr_position > 0)
  {
     fseek_status = fseek(infile,(ptr_position),SEEK_SET);
     ptr_position = ftell(infile);
  }
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\npointer position before reading the number of swapping rounds = "<<ptr_position;
  }
  
  fread_status = fread(&number_of_swapping_rounds,sizeof(number_of_swapping_rounds),1,infile);
  ptr_position =  ftell(infile);
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\nfread status after reading the number of swapping rounds = "<<fread_status;
    cout<<"\npointer position after reading the number of swapping rounds = "<<ptr_position; 
  }
  
  fclose(infile);
  
  /**
   * Reading seed for shuffling map_array
   */
  
  infile = fopen(config::constant_parameters_file_path,"rb");
  
  if(infile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for writing\nExiting...";
    exit(0);
  }
  
  
  if(ptr_position > 0)
  {
     fseek_status = fseek(infile,(ptr_position),SEEK_SET);
     ptr_position = ftell(infile);
  }
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\npointer position before reading seed = "<<ptr_position;
  }
  
  fread_status = fread(&seed,sizeof(seed),1,infile);
  ptr_position =  ftell(infile);
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\nfread status after reading seed = "<<fread_status;
    cout<<"\npointer position after reading seed = "<<ptr_position; 
  }
  
  fclose(infile);
  
  if(DEBUG_MAP_PARAMETERS == 1)
  {
    cout<<"\nnumber of rotation rounds = "<<number_of_rotation_rounds;
    cout<<"\nnumber of swapping rounds = "<<number_of_swapping_rounds;
    cout<<"\nseed = "<<seed;
  }
  
  /**
   * Generating and swapping chaotic map lut
   */
  pattern::MTSequence(map_array,15,1,15,seed);
  common::genMapLUTVec(map_choice_array,6);
  common::swapLUT(map_choice_array,map_array,6);
  
  /**
   * Chaotic map choice variables
   */
  config::ChaoticMap map_row_random_vec;
  config::ChaoticMap map_col_random_vec;
  config::ChaoticMap map_row_rotation_vec;
  config::ChaoticMap map_col_rotation_vec;
  config::ChaoticMap map_diffusion_array;
  
  /**
   * Parameter arrays
   */
  config::lm lm_parameters[6];
  config::lma lma_parameters[6];
  config::slmm slmm_parameters[6];
  config::lasm lasm_parameters[6];
  config::lalm lalm_parameters[6];
  config::mt mt_parameters[6];
  
  /**
   * Assigning chaotic map choices for each vector
   */
  map_row_random_vec = config::ChaoticMap(map_choice_array[0]);
  map_col_random_vec = config::ChaoticMap(map_choice_array[1]);
  map_row_rotation_vec = config::ChaoticMap(map_choice_array[2]);
  map_col_rotation_vec = config::ChaoticMap(map_choice_array[3]);
  map_diffusion_array = config::ChaoticMap(6);
  
  
  if(DEBUG_MAP_CHOICES_ARRAY == 1)
  {
    cout<<"\nMap choice array after reading = ";
    for(int i = 0; i < 6; ++i)
    {
      printf(" %d",map_choice_array[i]);
    }
  }
  
  /**
   * Reading map parameters
   */
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,infile,"rb",ptr_position,number_of_rotation_rounds);  
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,infile,"rb",ptr_position,number_of_rotation_rounds);
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,infile,"rb",ptr_position,number_of_swapping_rounds);
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,infile,"rb",ptr_position,number_of_swapping_rounds);
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,infile,"rb",ptr_position,1);
  
  if(DEBUG_MAP_PARAMETERS == 1)
  {
    /** 
     * Diplaying map parameters
     */
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rotation_rounds);
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rotation_rounds);
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_swapping_rounds);
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_swapping_rounds);  
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1);  
  }
  
  /** 
   * CPU vector declarations
   */
  uint8_t *enc_vec = (uint8_t*)calloc(total * channels,sizeof(uint8_t));
  uint8_t *dec_vec = (uint8_t*)calloc(total * channels,sizeof(uint8_t));
  uint8_t *final_vec = (uint8_t*)calloc(total * channels,sizeof(uint8_t));
  uint32_t *row_swap_lut_vec = (uint32_t*)calloc(m,sizeof(uint32_t));
  uint32_t *col_swap_lut_vec = (uint32_t*)calloc(n,sizeof(uint32_t));
  uint32_t *U = (uint32_t*)calloc(m,sizeof(uint32_t));
  uint32_t *V = (uint32_t*)calloc(n,sizeof(uint32_t));
  double *x = (double*)calloc(total * channels,sizeof(double));
  double *y = (double*)calloc(total * channels,sizeof(double));
  double *x_bar = (double*)calloc(total * channels,sizeof(double));
  double *y_bar = (double*)calloc(total * channels,sizeof(double));
  uint32_t *row_rotation_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *col_rotation_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *row_random_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *col_random_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  
  
  uint32_t *diffusion_array = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *dummy_lut_vec = (uint32_t*)calloc(2,sizeof(uint32_t));
  
  /** 
   * GPU Vector Declarations
   */
  uint8_t *gpu_enc_vec;
  uint8_t *gpu_dec_vec;
  uint8_t *gpu_final_vec;
  const uint32_t *gpu_row_swap_lut_vec;
  const uint32_t *gpu_col_swap_lut_vec;
  const uint32_t *gpu_U;
  const uint32_t *gpu_V;
  const uint32_t *gpu_diffusion_array;

  /**
   * Allocating GPU memory
   */
  cudaMalloc((void**)&gpu_enc_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_dec_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_final_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_col_swap_lut_vec,n * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_U,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_V,n * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_diffusion_array,total * channels * sizeof(uint32_t));
  
  
  /** 
   * Flattening 2D N X M image to 1D N X M vector
   */
  common::flattenImage(image,enc_vec,channels);
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\nencrypted image = ";
    common::printArray8(enc_vec,total * channels);
  }
  
  /**
   * Warming up kernel
   */
  dim3 warm_up_grid(1,1,1);
  dim3 warm_up_block(1,1,1);
  run_WarmUp(warm_up_grid,warm_up_block);
  
  /**
   * Undiffusion
   */
  if(DIFFUSION == 1)
  {
    pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,diffusion_array,dummy_lut_vec,map_diffusion_array,0,2,total * channels); 
      
    auto undiffusion_start = std::chrono::system_clock::now(); 
    serial::grayLevelTransform(enc_vec,diffusion_array,total * channels);
    
    auto undiffusion_end = std::chrono::system_clock::now();
    
    auto undiffusion_time = std::chrono::duration_cast<std::chrono::milliseconds>(undiffusion_end - undiffusion_start).count();
    cout<<"\nUndiffusion time =  "<<undiffusion_time<<" ms";
    
    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nUndiffused image = ");
      common::printArray8(enc_vec,total * channels);
    }
     
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      std::string undiffused_image = "";
      undiffused_image = image_name + "_undiffused_" + ".png";
      cv::Mat img_reshape(m,n,image.type(),enc_vec);
      bool undiffusion_status = cv::imwrite(undiffused_image,img_reshape);
      
      if(undiffusion_status == 1)
      {
        cout<<"\nUNDIFFUSION SUCCESSFUL";    
      }
      
      else
      {
        cout<<"\nUNDIFFUSION UNSUCCESSFUL";
      }
    }
  }  
  
  /**
   * Row and Column Unswapping
   */
  if(ROW_COL_SWAPPING == 1)
  {
    /** 
     * Display map parameters
     */
    for(i = number_of_swapping_rounds - 1; i >= 0; --i)
    {
      
      /**
       * Vector generation for row and coulumn swapping
       */
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_random_vec,row_swap_lut_vec,map_row_random_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_random_vec,col_swap_lut_vec,map_col_random_vec,i,n,total * channels);  
      
      dim3 dec_row_col_swap_grid(m,n,1);
      dim3 dec_row_col_swap_blocks(channels,1,1);
      
      /** 
       * Transferring CPU vectors to GPU memory
       */
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_dec_vec,dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_row_swap_lut_vec,row_swap_lut_vec,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_col_swap_lut_vec,col_swap_lut_vec,n * sizeof(uint32_t),cudaMemcpyHostToDevice);
      
      run_decRowColSwap(gpu_enc_vec,gpu_dec_vec,(const uint32_t*)gpu_row_swap_lut_vec,(const uint32_t*)gpu_col_swap_lut_vec,dec_row_col_swap_grid,dec_row_col_swap_blocks);
      
      /** 
       * Getting results from GPU memory
       */
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(dec_vec,gpu_dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      /**
       * Swapping
       */
      if(number_of_swapping_rounds > 1)
      {
        cudaMemcpy(enc_vec,dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToHost);
        cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      }
      
      if(DEBUG_VECTORS == 1)
      {
        
        cout<<"\n\ni = "<<i;
        printf("\nenc_vec = ");
        
        common::printArray8(enc_vec,total * channels);
        printf("\ndec_vec = ");
        
        common::printArray8(dec_vec,total * channels);
        cout<<"\nROUND = "<<i;
        
        printf("\nrow_swap_lut_vec = ");
        common::printArray32(row_swap_lut_vec,m);
        
        printf("\ncol_swap_lut_vec = ");
        common::printArray32(col_swap_lut_vec,n);
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        std::string unswapped_image = "";
        unswapped_image = image_name + "_unswapped" + "_ROUND_" + std::to_string(i + 1) + ".png";
        cv::Mat img_reshape(m,n,image.type(),dec_vec);
        bool swapping_status = cv::imwrite(unswapped_image,img_reshape);
        
        if(swapping_status == 1)
        {
          cout<<"\nUNSWAPPING SUCCESSFUL FOR ROUND "<<i + 1;
        }
        
        else
        {
          cout<<"UNSWAPPING UNSUCCESSFUL FOR ROUND "<<i + 1;
        }  
      }
   }
    

  } 
   
  /**
   * Row and Column Unrotation
   */
  
  if(ROW_COL_ROTATION == 1)
  {

    /**
     * Display map parameters
     */
    for(i = number_of_rotation_rounds - 1; i >= 0; --i)
    {
      
      cout<<"\nROUND "<<i;
      
      /** 
       * Vector generation for row and column unrotation
       */
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_rotation_vec,U,map_row_rotation_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_rotation_vec,V,map_col_rotation_vec,i,n,total * channels);
      
      dim3 dec_gen_cat_map_grid(m,n,1);
      dim3 dec_gen_cat_map_blocks(channels,1,1);
      
      /** 
       * Transferring CPU Vectors to GPU Memory
       */
      cudaMemcpy(gpu_dec_vec,dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_final_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_U,U,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_V,V,n * sizeof(uint32_t),cudaMemcpyHostToDevice);
      
      run_DecGenCatMap(gpu_dec_vec,gpu_final_vec,(const uint32_t*)gpu_V,(const uint32_t*)gpu_U,dec_gen_cat_map_grid,dec_gen_cat_map_blocks);
      
      /**
       * Getting results from GPU
       */
      cudaMemcpy(dec_vec,gpu_dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(final_vec,gpu_final_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      /**
       * Swapping
       */
      if(number_of_rotation_rounds > 1)
      {
        cudaMemcpy(dec_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToHost);
        cudaMemcpy(gpu_dec_vec,dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      }

      if(DEBUG_VECTORS == 1)
      {
        
        cout<<"\ni = "<<i;
        
        printf("\ndec_vec = ");
        common::printArray8(dec_vec,total * channels);
        
        printf("\nfinal_vec = ");
        common::printArray8(final_vec,total * channels);
        
        cout<<"\nU = ";
        common::printArray32(U,m);
        
        cout<<"\nV = ";
        common::printArray32(V,n);
        
        cout<<"\nRow rotation vec = ";
        common::printArray32(row_rotation_vec,total * channels);
        
        cout<<"\nColumn rotation vec = ";
        common::printArray32(col_rotation_vec,total * channels);
        
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        
        std::string unrotated_image = "";
        unrotated_image = image_name + "_unrotated" + "_ROUND_" + std::to_string(i + 1) + ".png";
        cv::Mat img_reshape(m,n,image.type(),final_vec);
        bool unrotation_status = cv::imwrite(unrotated_image,img_reshape);
        
        if(unrotation_status == 1)
        {
          cout<<"\nUNROTATION SUCCESSFUL FOR ROUND "<<i + 1;
        }
        
        else
        {
          cout<<"\nUNROTATION UNSUCCESSFUL FOR ROUND "<<i + 1;
        }
        
      }
      
      
    }  
  }
  
  auto end = std::chrono::system_clock::now();
  
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  cout<<"\nTotal decryption time = "<<elapsed / 1000000<<" s";
  
  
  std::string plain_image_path = std::string(argv[1]);
  std::string decrypted_image_path = image_name + "_unrotated" + "_ROUND_1" + ".png";
  
  char *constant_plain_image_path = const_cast<char*>(plain_image_path.c_str());
  char *constant_decrypted_image_path = const_cast<char*>(decrypted_image_path.c_str());
  
  /** 
   * Comparison of plain and decrypted images
   */
  cv::Mat plain_image = cv::imread(plain_image_path,cv::IMREAD_UNCHANGED);
  cv::Mat decrypted_image = cv::imread(decrypted_image_path,cv::IMREAD_UNCHANGED);
  bool decryption_status = common::checkImages(plain_image,decrypted_image);
  
  
  /** 
   * Compare SHA256 hashes of plain image and decrypted image
   */  
  std::string plain_image_hash = common::calc_sha256(constant_plain_image_path);
  std::string decrypted_image_hash = common::calc_sha256(constant_decrypted_image_path);
  bool hash_equality_status = 0;
  
  if(plain_image_hash == decrypted_image_hash)
  {
    hash_equality_status = 1;
  }  

  else
  {
    hash_equality_status = 0;
  }
  
  
  if(decryption_status == 0)
  {
    cout<<"\nDECRYPTION UNSUCCESSFUL\nHASHES DON'T MATCH AND DIFFERENCE IMAGE IS NONZERO";
  }
  
  else if(decryption_status == 1 and hash_equality_status == 1)
  {
    cout<<"\nDECRYPTION SUCCESSFUL\nHASHES EQUAL AS IMAGES HAVE BEEN MODIFIED THROUGH OPENCV";
  }
  
  else if(decryption_status == 1 and hash_equality_status == 0)
  {
     cout<<"\nDECRYPTION SUCCESSFUL\nHASHES UNEQUAL AS IMAGES HAVE NOT BEEN MODIFIED THROUGH OPENCV";
  }    
  
  else
  {
    cout<<"\nDECRYPTION UNSUCCESSFUL\nHASHES DON'T MATCH AND DIFFERENCE IMAGE IS NONZERO";
  }  
   
  
  return 0;
}

