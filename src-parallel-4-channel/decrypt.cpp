#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"
#include "include/io.hpp"

int main()
{
  cv::Mat image;
  
  
  image = cv::imread(config::diffused_image_path,cv::IMREAD_COLOR);
   
  if(!image.data)
  {
    cout<<"\nCould not open image from "<<config::diffused_image_path<<" \nExiting...";
    exit(0);
  }

  if(RESIZE_TO_DEBUG == 1)
  {
    cv::resize(image,image,cv::Size(config::cols,config::rows),CV_INTER_LANCZOS4);
  }
  
  uint32_t m = 0,n = 0,channels = 0, total = 0;
  
  
  double alpha = 0.00,beta = 0.00;
  int number_of_rounds = 0, i = 0;
  long ptr_position = 0; 
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  int fread_status = 9,fseek_status = 9;  

  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
  
  /*Reading number of rounds*/
  FILE *infile = fopen(config::constant_parameters_file_path,"rb");
  
  if(infile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for reading\nExiting...";
    exit(0);
  }
  
  cout<<"\npointer position before reading the number of rounds = "<<number_of_rounds;

  fread_status = fread(&number_of_rounds,sizeof(number_of_rounds),1,infile);
  ptr_position =  ftell(infile);
  cout<<"\npointer position after reading the number of rounds = "<<ptr_position; 
  fclose(infile);
  
  cout<<"\nNumber of rounds = "<<number_of_rounds;
  
  /*Chaotic map choice variables*/
  config::ChaoticMap map_row_random_vec;
  config::ChaoticMap map_col_random_vec;
  config::ChaoticMap map_row_rotation_vec;
  config::ChaoticMap map_col_rotation_vec;
  config::ChaoticMap map_diffusion_array;
  
  uint32_t *map_choice_array = (uint32_t*)calloc(5,sizeof(uint32_t));
  
  
  /*Reading map choice array*/
  infile = fopen("map_choices.bin","rb");
  if(infile == NULL)
  {
    printf("\nCould not open parameters.bin for reading map choices array\nExiting...");
    exit(0);
  }
  
  cout<<"\nfseek status before reading map choices array = "<<fseek_status;
  size_t size = 5 * sizeof(map_choice_array[0]);
  fread_status = fread(map_choice_array,size,1,infile);
  cout<<"\nfread status after reading map choices array = "<<fread_status;
  fclose(infile);
  
  /*Parameter arrays*/
  config::lm lm_parameters[number_of_rounds];
  config::lma lma_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::lalm lalm_parameters[number_of_rounds];
  config::mt mt_parameters[number_of_rounds];
  
  /*Assigning chaotic map choices for each vector*/
  map_row_random_vec = config::ChaoticMap(map_choice_array[0]);
  map_col_random_vec = config::ChaoticMap(map_choice_array[1]);
  map_row_rotation_vec = config::ChaoticMap(map_choice_array[2]);
  map_col_rotation_vec = config::ChaoticMap(map_choice_array[3]);
  map_diffusion_array = config::ChaoticMap(map_choice_array[4]);
  
  //if(DEBUG_VECTORS == 1)
  //{
    cout<<"\nMap choice array after reading = ";
    for(int i = 0; i < 5; ++i)
    {
      printf(" %d",int(map_choice_array[i]));
    }
  //}
  
  /*Reading map parameters*/
  ptr_position = pattern::readMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,infile,ptr_position,number_of_rounds);  
  ptr_position = pattern::readMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,infile,ptr_position,number_of_rounds);
  ptr_position = pattern::readMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,infile,ptr_position,number_of_rounds);
  ptr_position = pattern::readMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,infile,ptr_position,number_of_rounds);
  ptr_position = pattern::readMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,infile,ptr_position,1);
  
  /*Diplaying map parameters*/
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rounds);
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rounds);
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_rounds);
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_rounds);  
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1);  
  
  /*CPU vector declarations*/
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
  
  /*GPU Vector Declarations*/
  uint8_t *gpu_enc_vec;
  uint8_t *gpu_dec_vec;
  uint8_t *gpu_final_vec;
  const uint32_t *gpu_row_swap_lut_vec;
  const uint32_t *gpu_col_swap_lut_vec;
  const uint32_t *gpu_U;
  const uint32_t *gpu_V;
  const uint32_t *gpu_diffusion_array;

  //Allocating GPU memory
  cudaMalloc((void**)&gpu_enc_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_dec_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_final_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_col_swap_lut_vec,n * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_U,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_V,n * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_diffusion_array,total * channels * sizeof(uint32_t));
  
  
  /*Flattening image*/
  common::flattenImage(image,enc_vec,channels);
  //cout<<"\nencrypted image = ";
  //common::printArray8(enc_vec,total * channels);
  
  /*Warming up kernel*/
  dim3 warm_up_grid(1,1,1);
  dim3 warm_up_block(1,1,1);
  run_WarmUp(warm_up_grid,warm_up_block);
  
  if(DIFFUSION == 1)
  {
    cout<<"\nIn Undiffusion";
    
    /*Reading map parameters*/
    
    /*Display map parameters*/
    
    if(PARALLELIZED_DIFFUSION == 0)
    {
      long double diffusion_time = 0.00;
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,diffusion_array,dummy_lut_vec,map_diffusion_array,0,2,total * channels); 
      
      clock_t diffusion_start = std::clock(); 
      serial::grayLevelTransform(enc_vec,diffusion_array,total * channels);
      clock_t diffusion_end = std::clock();
      diffusion_time = 1000.0 * (diffusion_end - diffusion_start) / CLOCKS_PER_SEC ;
      printf("\nundiffusion time =  %Lf",diffusion_time);
    
    }
    
    else
    {
     
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,diffusion_array,dummy_lut_vec,map_diffusion_array,0,2,total * channels); 
      
      //Transferring CPU vectors to GPU memory
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_diffusion_array,diffusion_array,total * channels * sizeof(uint32_t),cudaMemcpyHostToDevice);
      
      dim3 grid_gray_level_transform(m * n,1,1);
      dim3 block_gray_level_transform(channels,1,1);
      
      run_grayLevelTransform(gpu_enc_vec,(const uint32_t*)gpu_diffusion_array,grid_gray_level_transform,block_gray_level_transform);
      
      //Getting results from GPU memory 
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
    }

    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nUndiffused image = ");
      common::printArray8(enc_vec,total * channels);
    }
     
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      cv::Mat img_reshape(m,n,CV_8UC3,enc_vec);
      cv::imwrite(config::undiffused_image_path,img_reshape);
    }
    

  }  

  if(ROW_COL_SWAPPING == 1)
  {
    /*Row and Column Unswapping*/
    cout<<"\nIn row and column unswapping";
    
    /*Reading map parameters*/
    
    /*Display map parameters*/
    for(i = number_of_rounds - 1; i >= 0; --i)
    {
      
      /*Vector generation for row and coulumn swapping*/
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_random_vec,row_swap_lut_vec,map_row_random_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_random_vec,col_swap_lut_vec,map_col_random_vec,i,n,total * channels);  
      
      dim3 dec_row_col_swap_grid(m,n,1);
      dim3 dec_row_col_swap_blocks(channels,1,1);
      
      //Transferring CPU vectors to GPU memory
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_dec_vec,dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_row_swap_lut_vec,row_swap_lut_vec,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_col_swap_lut_vec,col_swap_lut_vec,n * sizeof(uint32_t),cudaMemcpyHostToDevice);
      
      run_decRowColSwap(gpu_enc_vec,gpu_dec_vec,(const uint32_t*)gpu_row_swap_lut_vec,(const uint32_t*)gpu_col_swap_lut_vec,dec_row_col_swap_grid,dec_row_col_swap_blocks);
      
      //Getting results from GPU memory
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(dec_vec,gpu_dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      //Swapping
      if(number_of_rounds > 1)
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
        /*cout<<"\n\ni = "<<i;
        printf("\nrow_swap_lut_vec = ");
        common::printArray32(row_swap_lut_vec,m);
        printf("\ncol_swap_lut_vec = ");
        common::printArray32(col_swap_lut_vec,n);*/
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        config::unswapped_image = "";
        config::unswapped_image = config::image_name + "_unswapped" + "_ROUND_" + std::to_string(i) + ".png";
        cout<<"\n"<<config::unswapped_image;
        //printf("\nindex = %d",cnt);
        cv::Mat img_reshape(m,n,CV_8UC3,dec_vec);
        cv::imwrite(config::unswapped_image,img_reshape);  
      }
   }
    

  } 
  
  if(ROW_COL_ROTATION == 1)
  {
    /*Row and Column Unrotation*/
    cout<<"\nIn row and column unrotation ";
    

    
    /*Display map parameters*/
    
    
    for(i = number_of_rounds - 1; i >= 0; --i)
    {
      
      cout<<"\nROUND "<<i;
      
      /*Vector generation for row and column unrotation*/
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_rotation_vec,U,map_row_rotation_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_rotation_vec,V,map_col_rotation_vec,i,n,total * channels);
      
      dim3 dec_gen_cat_map_grid(m,n,1);
      dim3 dec_gen_cat_map_blocks(channels,1,1);
      
      /*Transferring CPU Vectors to GPU Memory*/
      cudaMemcpy(gpu_dec_vec,dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_final_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_U,U,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_V,V,n * sizeof(uint32_t),cudaMemcpyHostToDevice);
      
      run_DecGenCatMap(gpu_dec_vec,gpu_final_vec,(const uint32_t*)gpu_V,(const uint32_t*)gpu_U,dec_gen_cat_map_grid,dec_gen_cat_map_blocks);
      
      //Getting results from GPU
      cudaMemcpy(dec_vec,gpu_dec_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(final_vec,gpu_final_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      //Swapping
      if(number_of_rounds > 1)
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
        /*cout<<"\nU = ";
        common::printArray32(U,m);
        cout<<"\nV = ";
        common::printArray32(V,n);*/
        /*cout<<"\nRow rotation vec = ";
        common::printArray32(row_rotation_vec,total * channels);
        cout<<"\nColumn rotation vec = ";
        common::printArray32(col_rotation_vec,total * channels);
        */
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        //printf("\nindex = %d",cnt);
        config::unrotated_image = "";
        config::unrotated_image = config::image_name + "_unrotated" + "_ROUND_" + std::to_string(i) + config::extension;
        cout<<"\n"<<config::unrotated_image;
        cv::Mat img_reshape(m,n,CV_8UC3,final_vec);
        int imwrite_status = cv::imwrite(config::unrotated_image,img_reshape);
        cout<<"\nimwrite_status = "<<imwrite_status;
      }
      
      //Swapping input and output vectors
    }  
  }
 
  return 0;
}

