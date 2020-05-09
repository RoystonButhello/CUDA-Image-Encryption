#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"
#include "include/io.hpp"


int main()
{
  cv::Mat image;
  image = cv::imread(config::input_image_path,cv::IMREAD_COLOR);
  
  if(!image.data)
  {
    cout<<"\nCould not open image from "<<config::input_image_path<<" \nExiting...";
    exit(0);
  }

  if(RESIZE_TO_DEBUG == 1)
  {
    cv::resize(image,image,cv::Size(config::cols,config::rows),CV_INTER_LANCZOS4);
  }
  
  
  uint32_t m = 0,n = 0,channels = 0, total = 0;
  //int number_of_rounds = 2;
  int number_of_rounds = 3;
  int fseek_status = 9,fwrite_status = 9,i = 0;
  
  long ptr_position = 0;
  double alpha = 0.00,beta = 0.00;
  
  cudaEvent_t start_rotate,stop_rotate,start_swap,stop_swap;


  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;

  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
  /*Chaotic map choice variables*/
  config::ChaoticMap map_row_random_vec;
  config::ChaoticMap map_col_random_vec;
  config::ChaoticMap map_row_rotation_vec;
  config::ChaoticMap map_col_rotation_vec;
  config::ChaoticMap map_diffusion_array;
  
  
  
  
  /*Parameter arrays*/
  config::lm lm_parameters[number_of_rounds];
  config::lma lma_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::lalm lalm_parameters[number_of_rounds];
  config::mt mt_parameters[number_of_rounds];
  
  /*CPU vector declarations*/
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
  uint32_t *map_choice_array = (uint32_t*)calloc(5,sizeof(uint32_t));
  
  /*GPU vector declarations*/
  uint8_t *gpu_img_vec;
  uint8_t *gpu_enc_vec;
  uint8_t *gpu_final_vec;
  const uint32_t *gpu_U;
  const uint32_t *gpu_V;
  const uint32_t *gpu_row_swap_lut_vec;
  const uint32_t *gpu_col_swap_lut_vec;
  const uint32_t *gpu_diffusion_array;

  /*Allocating GPU memory*/
  cudaMalloc((void**)&gpu_img_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_enc_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_final_vec,total * channels * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_U,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_V,n * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMalloc((void**)&gpu_col_swap_lut_vec,n * sizeof(uint32_t));
  
  
  
  /*Flattening image*/
  common::flattenImage(image,img_vec,channels);
  //cout<<"\nplain image = ";
  //common::printArray8(img_vec,total * channels);
  
  /*Generating and swapping chaotic map lut*/
  pattern::MTSequence(map_array,15,1,15,1000);
  common::genMapLUTVec(map_choice_array,5);
  common::swapLUT(map_choice_array,map_array,5);
  
  /*Assigning chaotic map choices for each vector*/
  map_row_random_vec = config::ChaoticMap(map_choice_array[0]);
  map_col_random_vec = config::ChaoticMap(map_choice_array[1]);
  map_row_rotation_vec = config::ChaoticMap(map_choice_array[2]);
  map_col_rotation_vec = config::ChaoticMap(map_choice_array[3]);
  map_diffusion_array = config::ChaoticMap(map_choice_array[4]);
  
  /*Storing each map choice into an array for writing
  map_choice_array[0] = int(map_row_random_vec);
  map_choice_array[1] = int(map_col_random_vec);
  map_choice_array[2] = int(map_row_rotation_vec);
  map_choice_array[3] = int(map_col_rotation_vec);
  map_choice_array[4] = int(map_diffusion_array);*/
  
  //if(DEBUG_VECTORS == 1)
  //{
    cout<<"\nMap choices = ";
    for(int i = 0 ; i < 5; ++i)
    {
      printf(" %d",int(map_choice_array[i]));
    }
  //}
  
 
    /*Initializing map parameters*/
    pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rounds);
    pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rounds);
    pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_rounds);
    pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_rounds);
    pattern::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1); 
    
    /*Assigning map parameters*/
    pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rounds);
    pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rounds);
    pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_rounds);
    pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_rounds);
    pattern::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1);
   
  /*Writing number of rounds*/
  FILE *outfile = fopen(config::constant_parameters_file_path,"wb");
  
  if(outfile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for writing\nExiting...";
    exit(0);
  }
  
  cout<<"\npointer position before writing the number of rounds = "<<number_of_rounds;

  fwrite_status = fwrite(&number_of_rounds,sizeof(number_of_rounds),1,outfile);
  ptr_position =  ftell(outfile);
  cout<<"\nfwrite status after writing the number of rounds = "<<fwrite_status;
  cout<<"\npointer position after writing the number of rounds = "<<ptr_position; 
  fclose(outfile);
  
  cout<<"\nNumber of rounds = "<<number_of_rounds;
  
  /*Writing map choices array*/
  outfile = fopen("map_choices.bin","wb");
  if(outfile == NULL)
  {
    printf("\nCould not open parameters.bin for writing map choices array\nExiting...");
    exit(0);
  }
  
  //cout<<"\npointer position before writing map choices array = "<<ptr_position;
  
  
  cout<<"\nfseek status before writing map choices array = "<<fseek_status;
  //cout<<"\npointer position before wriitng map choices array = "<<ptr_position;
  size_t size = 5 * sizeof(map_choice_array[0]);
  fwrite_status = fwrite(map_choice_array,size,1,outfile);
  cout<<"\nfwrite status after writing map choices array = "<<fwrite_status;
  fclose(outfile); 
  
  /*Writing map parameters*/
  ptr_position = pattern::writeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,outfile,ptr_position,number_of_rounds);  
  ptr_position = pattern::writeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,outfile,ptr_position,number_of_rounds);
  ptr_position = pattern::writeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,outfile,ptr_position,number_of_rounds);
  ptr_position = pattern::writeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,outfile,ptr_position,number_of_rounds);
  ptr_position = pattern::writeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,outfile,ptr_position,1);
    
  /*Display map parameters*/
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_rotation_vec,number_of_rounds);
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_rotation_vec,number_of_rounds);
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_row_random_vec,number_of_rounds);
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_col_random_vec,number_of_rounds);
  pattern::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,map_diffusion_array,1);
  
  /*Warming up GPU*/
  dim3 warm_up_grid(1,1,1);
  dim3 warm_up_block(1,1,1);
  run_WarmUp(warm_up_grid,warm_up_block);
  
  if(ROW_COL_ROTATION == 1)
  {
    cout<<"\nIn row and column rotation";
        
    for(i = 0; i < number_of_rounds; ++i)
    {
      cout<<"\nROUND "<<i;
      
      /*Vector generation for row and column rotation*/
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_rotation_vec,U,map_row_rotation_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_rotation_vec,V,map_col_rotation_vec,i,n,total * channels);   
      
      /*Row and Column Rotation*/
      dim3 enc_gen_cat_map_grid(m,n,1);
      dim3 enc_gen_cat_map_blocks(channels,1,1);
      
      //Copying CPU vectors to GPU memory
      cudaMemcpy(gpu_img_vec,img_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_U,U,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_V,V,n * sizeof(uint32_t),cudaMemcpyHostToDevice);
        
      run_EncGenCatMap(gpu_img_vec,gpu_enc_vec,(const uint32_t*)gpu_V,(const uint32_t*)gpu_U,enc_gen_cat_map_grid,enc_gen_cat_map_blocks);
      
      //Getting results from GPU
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(img_vec,gpu_img_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      //Swapping
      if(number_of_rounds > 1)
      {  
        cudaMemcpy(img_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToHost);
        cudaMemcpy(gpu_img_vec,img_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      }
      
      
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        config::rotated_image = "";
        config::rotated_image = config::image_name + "_rotated" + "_ROUND_" + std::to_string(i) + config::extension;
        
        //cout<<"\n"<<config::rotated_image;
        cv::Mat img_reshape(m,n,CV_8UC3,enc_vec);
        cv::imwrite(config::rotated_image,img_reshape);
        
      }
      
      
      if(DEBUG_VECTORS == 1)
      {
        
        cout<<"\nimg_vec = ";
        common::printArray8(img_vec,total * channels);
        printf("\nenc_vec = ");
        common::printArray8(enc_vec,total * channels);
        /*cout<<"\nU = ";
        common::printArray32(U,m);
        cout<<"\nV = ";
        common::printArray32(V,n);*/
        /*cout<<"\nRow rotation vec = ";
        common::printArray32(row_rotation_vec,total * channels);
        cout<<"\nColumn rotation vec = ";
        common::printArray32(col_rotation_vec,total * channels);*/
      }
      
      //Swapping input and output vectors
      
    }
 } 
  
  if(ROW_COL_SWAPPING == 1)
  {    
    /*Row and Column Swapping*/
    cout<<"\nIn row and column swapping";
    for(i = 0; i < number_of_rounds; ++i)
    {
      /*Vector generation for row and coulumn swapping*/
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,row_random_vec,row_swap_lut_vec,map_row_random_vec,i,m,total * channels);
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,col_random_vec,col_swap_lut_vec,map_col_random_vec,i,n,total * channels);   
      
      dim3 enc_row_col_swap_grid(m,n,1);
      dim3 enc_row_col_swap_blocks(channels,1,1);
      
      //Transferring vectors from CPU to GPU memory
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_final_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_row_swap_lut_vec,row_swap_lut_vec,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_col_swap_lut_vec,col_swap_lut_vec,n * sizeof(uint32_t),cudaMemcpyHostToDevice);      

      run_encRowColSwap(gpu_enc_vec,gpu_final_vec,(const uint32_t *)gpu_row_swap_lut_vec,(const uint32_t*)gpu_col_swap_lut_vec,enc_row_col_swap_grid,enc_row_col_swap_blocks);
      
      //Getting results from GPU
      cudaMemcpy(final_vec,gpu_final_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      //Swapping
      if(number_of_rounds  > 1)
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
        /*cout<<"\n\ni = "<<i;
        printf("\nrow_swap_lut_vec = ");
        common::printArray32(row_swap_lut_vec,m);
        printf("\ncol_swap_lut_vec = ");
        common::printArray32(col_swap_lut_vec,n);*/
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        config::swapped_image = "";
        config::swapped_image = config::image_name + "_swapped" + "_ROUND_" + std::to_string(i) + config::extension;
        cout<<"\n"<<config::swapped_image;  
        cv::Mat img_reshape(m,n,CV_8UC3,final_vec);
        cv::imwrite(config::swapped_image,img_reshape); 
      }
      
    }
  }
  
  if(DIFFUSION == 1)
  {
    cout<<"\nIn Diffusion";
    /*Display map parameters*/
    
    if(PARALLELIZED_DIFFUSION == 0)
    {
      long double diffusion_time = 0.00;
      /*Generating diffusion array*/
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,diffusion_array,dummy_lut_vec,map_diffusion_array,0,2,total * channels);  
      
      std::clock_t diffusion_start = std::clock(); 
      serial::grayLevelTransform(final_vec,diffusion_array,total * channels);
      std::clock_t diffusion_end = std::clock();
      diffusion_time = 1000.0 * (diffusion_end - diffusion_start) / CLOCKS_PER_SEC ;
      printf("\ndiffusion time =  %Lf",diffusion_time);
    }
    
    else
    {
      /*Generating diffusion array*/
      pattern::selectChaoticMap(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,x,y,x_bar,y_bar,diffusion_array,dummy_lut_vec,map_diffusion_array,0,2,total * channels);  
      
      //Allocating GPU memory
      cudaMalloc((void**)&gpu_diffusion_array,total * channels * sizeof(uint32_t));
      
      
      //Copying vectors to GPU memory
      cudaMemcpy(gpu_final_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_diffusion_array,diffusion_array,total * channels * sizeof(uint32_t),cudaMemcpyHostToDevice);
      
      dim3 grid_gray_level_transform(m * n,1,1);
      dim3 block_gray_level_transform(channels,1,1);
      
      run_grayLevelTransform(gpu_final_vec,(const uint32_t*)gpu_diffusion_array,grid_gray_level_transform,block_gray_level_transform);
      
      //Getting results from GPU memory to CPU
      cudaMemcpy(final_vec,gpu_final_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
    }
    
    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nRotated Swapped and Diffused image = ");
      common::printArray8(final_vec,total * channels);
    }
     
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      cv::Mat img_reshape(m,n,CV_8UC3,final_vec);
      cv::imwrite(config::diffused_image_path,img_reshape);
    }
    
  }
  return 0;
}

