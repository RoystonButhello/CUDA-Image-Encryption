#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"
#include "include/io.hpp"


int main(int argc,char *argv[])
{

  /**
   * Get file path
   */
  std::string input_image_path = std::string(argv[1]);
  cout<<"\nInput image = "<<input_image_path;
  
  
  if(input_image_path == "")
  {
    cout<<"\nNo image name specified.Please specify an image name.\nExiting...";
    exit(0);
  }
  
  /**
   * Get image name from file path
   */
  std::string image_name = common::getFileNameFromPath(input_image_path);
  
  auto start = std::chrono::system_clock::now();
  
  cv::Mat image;
  image = cv::imread(input_image_path,cv::IMREAD_UNCHANGED);
  
  if(!image.data)
  {
    cout<<"\nCould not open image from "<<input_image_path<<" \nExiting...";
    exit(0);
  }
  
  uint32_t m = 0,n = 0,channels = 0, total = 0;
  int number_of_rotation_rounds = 0;
  int number_of_swapping_rounds = 0;
  int seed = 0;
  int fseek_status = 9,fwrite_status = 9,i = 0;
  

  long ptr_position = 0;
  double alpha = 0.00,beta = 0.00;
  
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;

  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
      
 /**
  * Assign the number of rotation and swapping rounds
  */
  number_of_rotation_rounds = common::getRandomInteger(5,5);
  number_of_swapping_rounds = common::getRandomInteger(5,5);
   

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
   * CPU vector declarations
   */
  uint8_t *img_vec = (uint8_t*)calloc(total * channels,sizeof(uint8_t));
  uint8_t *enc_vec = (uint8_t*)calloc(total * channels,sizeof(uint8_t));
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
  uint32_t *map_array = (uint32_t*)calloc(15,sizeof(uint32_t));
  uint32_t *map_choice_array = (uint32_t*)calloc(6,sizeof(uint32_t));
  
  
  /**
   * GPU vector declarations
   */
  uint8_t *gpu_img_vec;
  uint8_t *gpu_enc_vec;
  uint8_t *gpu_final_vec;
  const uint32_t *gpu_U;
  const uint32_t *gpu_V;
  const uint32_t *gpu_row_swap_lut_vec;
  const uint32_t *gpu_col_swap_lut_vec;
  const uint32_t *gpu_diffusion_array;

  /**
   * Allocating GPU memory
   */
  cudaMalloc((void**)&gpu_img_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_enc_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_final_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_U,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_V,n * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_col_swap_lut_vec,n * sizeof(uint32_t));
     
  /** 
   * Flattening 2D N X M image to 1D N X M vector
   */
  common::flattenImage(image,img_vec,channels);
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\nplain image = ";
    common::printArray8(img_vec,total * channels);
  }

  /**
   * Generating and swapping chaotic map lut
   */
  seed = common::getRandomInteger(1000,2000);
  pattern::MTSequence(map_array,15,1,15,seed);
  common::genMapLUTVec(map_choice_array,6);
  common::swapLUT(map_choice_array,map_array,6);
  
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
    cout<<"\nMap choices = ";
    for(int i = 0 ; i < 6; ++i)
    {
      printf(" %d",map_choice_array[i]);
    }
  }
  

  /**
   * Initializing map parameters
   */
  pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rotation_rounds);
  pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rotation_rounds);
  pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_swapping_rounds);
  pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_swapping_rounds);
  pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1); 
  
  /** 
   * Assigning map parameters
   */
  pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rotation_rounds);
  pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rotation_rounds);
  pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_swapping_rounds);
  pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_swapping_rounds);
  pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1);

  /**
   * Writing number of rotation rounds
   */
  FILE *outfile = fopen(config::constant_parameters_file_path,"wb");
  
  if(outfile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for writing\nExiting...";
    exit(0);
  }
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\npointer position before writing the number of rotation rounds = "<<ptr_position;
  }
  
  fwrite_status = fwrite(&number_of_rotation_rounds,sizeof(number_of_rotation_rounds),1,outfile);
  ptr_position =  ftell(outfile);
  
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\nfwrite status after writing the number of rotation rounds = "<<fwrite_status;
    cout<<"\npointer position after writing the number of rotation rounds = "<<ptr_position; 
  }
  
  
  fclose(outfile);
  
  /**
   * Writing number of swapping rounds
   */
  outfile = fopen(config::constant_parameters_file_path,"ab");
  
  if(outfile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for writing\nExiting...";
    exit(0);
  }
  
  if(ptr_position > 0)
  {
     fseek_status = fseek(outfile,(ptr_position + 1),SEEK_SET);
     ptr_position = ftell(outfile);
  }
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\npointer position before writing the number of swapping rounds = "<<ptr_position;
  }
  
  fwrite_status = fwrite(&number_of_swapping_rounds,sizeof(number_of_swapping_rounds),1,outfile);
  ptr_position =  ftell(outfile);
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\nfwrite status after writing the number of swapping rounds = "<<fwrite_status;
    cout<<"\npointer position after writing the number of swapping rounds = "<<ptr_position; 
  }
  
  fclose(outfile);
  
  
  /**
   * Writing seed used for shuffling the map_array
   */
  outfile = fopen(config::constant_parameters_file_path,"ab");
  
  if(outfile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for writing\nExiting...";
    exit(0);
  }
  
  if(ptr_position > 0)
  {
     fseek_status = fseek(outfile,(ptr_position + 1),SEEK_SET);
     ptr_position = ftell(outfile);
  }
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\npointer position before writing seed = "<<ptr_position;
  }
 
  fwrite_status = fwrite(&seed,sizeof(seed),1,outfile);
  ptr_position =  ftell(outfile);
  
  if(DEBUG_READ_WRITE == 1)
  {
    cout<<"\nfwrite status after writing seed = "<<fwrite_status;
    cout<<"\npointer position after writing seed = "<<ptr_position; 
  }
  
  fclose(outfile);
  
  if(DEBUG_MAP_PARAMETERS == 1)
  {
    cout<<"\nnumber of rotation rounds = "<<number_of_rotation_rounds;
    cout<<"\nnumber of swapping rounds = "<<number_of_swapping_rounds;
    cout<<"\nseed = "<<seed;
  }
  
  
  /**
   * Writing map parameters
   */
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,outfile,"ab",ptr_position,number_of_rotation_rounds);  
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,outfile,"ab",ptr_position,number_of_rotation_rounds);
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,outfile,"ab",ptr_position,number_of_swapping_rounds);
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,outfile,"ab",ptr_position,number_of_swapping_rounds);
  ptr_position = pattern::rwMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,outfile,"ab",ptr_position,1);
  

  if(DEBUG_MAP_PARAMETERS == 1)
  {  
    /**
     * Display map parameters
     */
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rotation_rounds);
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rotation_rounds);
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_swapping_rounds);
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_swapping_rounds);
    pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1);
  }
  
  /**
   * Warming up GPU
   */
  dim3 warm_up_grid(1,1,1);
  dim3 warm_up_block(1,1,1);
  run_WarmUp(warm_up_grid,warm_up_block);
  
  if(ROW_COL_ROTATION == 1)
  {
    /**
     * Row and Column Rotation
     */
    
    for(i = 0; i < number_of_rotation_rounds; ++i)
    {
      /**
       * Vector generation for row and column rotation
       */
      
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_rotation_vec,U,map_row_rotation_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_rotation_vec,V,map_col_rotation_vec,i,n,total * channels);   
    
      /**
       * Row and Column Rotation
       */
      dim3 enc_gen_cat_map_grid(m,n,1);
      dim3 enc_gen_cat_map_blocks(channels,1,1);
      
      /**
       * Copying CPU vectors to GPU memory
       */
      cudaMemcpy(gpu_img_vec,img_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_U,U,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_V,V,n * sizeof(uint32_t),cudaMemcpyHostToDevice);
        
      run_EncGenCatMap(gpu_img_vec,gpu_enc_vec,(const uint32_t*)gpu_V,(const uint32_t*)gpu_U,enc_gen_cat_map_grid,enc_gen_cat_map_blocks);
      
      /**
       * Getting results from GPU
       */
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(img_vec,gpu_img_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      /**
       * Swapping img_vec and enc_vec
       */
      if(number_of_rotation_rounds > 1)
      {  
        cudaMemcpy(img_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToHost);
        cudaMemcpy(gpu_img_vec,img_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      }
      
      
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        std::string rotated_image = "";
        rotated_image = image_name + "_rotated" + "_ROUND_" + std::to_string(i + 1) + ".png";
        cv::Mat img_reshape(m,n,image.type(),enc_vec);
        bool rotate_status = cv::imwrite(rotated_image,img_reshape);
        
        if(rotate_status == 1)
        {
          cout<<"\nROTATION SUCCESSFUL FOR ROUND "<<i + 1;  
        }
        
        else
        {
          cout<<"\nROTATION UNSUCCESSFUL FOR ROUND "<<i + 1;
        }
      }
      
      if(DEBUG_VECTORS == 1)
      {
        
        cout<<"\nimg_vec = ";
        common::printArray8(img_vec,total * channels);
        
        printf("\nenc_vec = ");
        common::printArray8(enc_vec,total * channels);
        
        cout<<"\nU = ";
        common::printArray32(U,m);
        
        cout<<"\nV = ";
        common::printArray32(V,n);
        
        cout<<"\nRow rotation vec = ";
        common::printArray32(row_rotation_vec,total * channels);
        
        cout<<"\nColumn rotation vec = ";
        common::printArray32(col_rotation_vec,total * channels);
      }
      
      
      
    }
 } 
  
  if(ROW_COL_SWAPPING == 1)
  {    
    /**
     * Row and Column Swapping
     */
    
    for(i = 0; i < number_of_swapping_rounds; ++i)
    {
      /**
       * Vector generation for row and coulumn swapping
       */
     
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_random_vec,row_swap_lut_vec,map_row_random_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_random_vec,col_swap_lut_vec,map_col_random_vec,i,n,total * channels);   
      
      dim3 enc_row_col_swap_grid(m,n,1);
      dim3 enc_row_col_swap_blocks(channels,1,1);
      
      /**
       * Transferring vectors from CPU to GPU memory
       */
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_final_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_row_swap_lut_vec,row_swap_lut_vec,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_col_swap_lut_vec,col_swap_lut_vec,n * sizeof(uint32_t),cudaMemcpyHostToDevice);      

      run_encRowColSwap(gpu_enc_vec,gpu_final_vec,(const uint32_t *)gpu_row_swap_lut_vec,(const uint32_t*)gpu_col_swap_lut_vec,enc_row_col_swap_grid,enc_row_col_swap_blocks);
      
      /**
       * Getting results from GPU
       */
      cudaMemcpy(final_vec,gpu_final_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      /**
       * Swapping enc_vec and final_vec
       */
      if(number_of_swapping_rounds > 1)
      {
        cudaMemcpy(enc_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToHost);
        cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      }
      
      
      if(DEBUG_VECTORS == 1)
      {
        cout<<"\n\ni = "<<i;
        printf("\nenc_vec = ");
        
        common::printArray8(enc_vec,total * channels);
        printf("\nfinal_vec = ");
        
        common::printArray8(final_vec,total * channels);
        cout<<"\n\ni = "<<i;
        
        printf("\nrow_swap_lut_vec = ");
        common::printArray32(row_swap_lut_vec,m);
        
        printf("\ncol_swap_lut_vec = ");
        common::printArray32(col_swap_lut_vec,n);
      
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        
        std::string swapped_image = "";
        swapped_image = image_name + "_swapped" + "_ROUND_" + std::to_string(i + 1) + ".png";
        cv::Mat img_reshape(m,n,image.type(),final_vec);
        bool swap_status = cv::imwrite(swapped_image,img_reshape);
        
        if(swap_status == 1)
        {
          cout<<"\nSWAPPING SUCCESSFUL FOR "<<"ROUND "<<i + 1;  
        }
        
        else
        {
          cout<<"\nSWAPPING UNSUCCESSFUL FOR "<<"ROUND "<<i + 1;  
        }
         
      }
      
    }
  }
  
  /**
   * Diffusion
   */
  
  if(DIFFUSION == 1)
  {
    
      /**
       * Generating diffusion array
       */
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,diffusion_array,dummy_lut_vec,map_diffusion_array,0,2,total * channels);  
      
      auto diffusion_start = std::chrono::system_clock::now(); 
      
      serial::grayLevelTransform(final_vec,diffusion_array,total * channels);
      
      auto diffusion_end = std::chrono::system_clock::now();
      
      auto diffusion_time = std::chrono::duration_cast<std::chrono::milliseconds>(diffusion_end - diffusion_start).count();
      cout<<"\nDiffusion time ="<<diffusion_time <<" ms";
    
    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nRotated Swapped and Diffused image = ");
      common::printArray8(final_vec,total * channels);
    }
     
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      std::string diffused_image  = "";
      diffused_image = image_name + "_diffused" + ".png";
      cv::Mat img_reshape(m,n,image.type(),final_vec);
      bool diffusion_status = cv::imwrite(diffused_image,img_reshape);
      
      if(diffusion_status == 1)
      {
        cout<<"\nDIFFUSION SUCCESSFUL";
          
      }
      
      else
      {
        cout<<"\nDIFFUSION UNSUCCESSFUL";  
      }
      
    }
    
  }
   auto end = std::chrono::system_clock::now();
   
   auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
   
   cout<<"\nTotal encryption time  = "<<elapsed / 1000000<<" s";
   
   return 0;
}


