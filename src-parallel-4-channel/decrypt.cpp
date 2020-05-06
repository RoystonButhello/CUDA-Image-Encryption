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
  int number_of_rounds = 0;
  long ptr_position = 0; 
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  int fread_status = 9,fseek_status = 9;  

  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
  
  /*Reading parameters*/
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
  
  /*Parameter arrays*/
  config::lm lm_parameters[number_of_rounds];
  config::lma lma_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::lalm lalm_parameters[number_of_rounds];
  config::mt mt_parameters[0];
  
  ptr_position = readLMParameters(infile,config::constant_parameters_file_path,"rb",lm_parameters,0,number_of_rounds,ptr_position);
  ptr_position = readLMAParameters(infile,config::constant_parameters_file_path,"rb",lma_parameters,0,number_of_rounds,ptr_position);
  ptr_position = readSLMMParameters(infile,config::constant_parameters_file_path,"rb",slmm_parameters,0,number_of_rounds,ptr_position);
  ptr_position = readLASMParameters(infile,config::constant_parameters_file_path,"rb",lasm_parameters,0,number_of_rounds,ptr_position);
  ptr_position = readMTParameters(infile,config::constant_parameters_file_path,"rb",mt_parameters,0,1,ptr_position);
  
  /*Display parameters after reading*/
  //common::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
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
  uint32_t *row_rotation_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *col_rotation_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *row_random_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *col_random_vec = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  uint32_t *diffusion_array = (uint32_t*)calloc(total * channels,sizeof(uint32_t));
  
  /*GPU Vector Declarations*/
  uint8_t *gpu_enc_vec;
  uint8_t *gpu_dec_vec;
  uint8_t *gpu_final_vec;
  const uint32_t *gpu_row_swap_lut_vec;
  const uint32_t *gpu_col_swap_lut_vec;
  const uint32_t *gpu_U;
  const uint32_t *gpu_V;
  const uint32_t *gpu_diffusion_array;

  
  
  
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
    
    if(PARALLELIZED_DIFFUSION == 0)
    {
      long double diffusion_time = 0.00;
      pattern::MTSequence(diffusion_array,total * channels,config::lower_limit,config::upper_limit,mt_parameters[0].seed_1);
      
      clock_t diffusion_start = std::clock(); 
      serial::grayLevelTransform(final_vec,diffusion_array,total * channels);
      clock_t diffusion_end = std::clock();
      diffusion_time = 1000.0 * ((diffusion_end - diffusion_start) / CLOCKS_PER_SEC );
      cout<<"\nundiffusion time = "<<diffusion_time<<" ms";
    
    }
    
    else
    {
      pattern::MTSequence(diffusion_array,total * channels,config::lower_limit,config::upper_limit,mt_parameters[0].seed_1);
      
      //Allocating GPU memory
      cudaMalloc((void**)&gpu_enc_vec,total * channels * sizeof(uint32_t));
      cudaMalloc((void**)&gpu_diffusion_array,total * channels * sizeof(uint32_t));
      
      //Transferring CPU vectors to GPU memory
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)gpu_diffusion_array,diffusion_array,total * channels * sizeof(uint32_t),cudaMemcpyHostToDevice);
      
      dim3 grid_gray_level_transform(m * n,1,1);
      dim3 block_gray_level_transform(channels,1,1);
      
      run_grayLevelTransform(gpu_enc_vec,(const uint32_t*)gpu_diffusion_array,grid_gray_level_transform,block_gray_level_transform);
      
      //Getting results from GPU memory 
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      //Free GPU memory
      cudaFree((void*)gpu_enc_vec);
      cudaFree((void*)gpu_diffusion_array);
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
    for(int i = number_of_rounds - 1; i >= 0; --i)
    {
      
      /*Vector generation for row and coulumn swapping*/
      x[0] = slmm_parameters[i].x_init;
      y[0] = slmm_parameters[i].y_init;
      pattern::twodSineLogisticModulationMap(x,y,row_random_vec,slmm_parameters[i].alpha,slmm_parameters[i].beta,total * channels);
      
      x[0] = lasm_parameters[i].x_init;
      y[0] = lasm_parameters[i].y_init;
      pattern::twodLogisticAdjustedSineMap(x,y,col_random_vec,lasm_parameters[i].myu,total * channels);
      
      common::genLUTVec(row_swap_lut_vec,m);
      common::genLUTVec(col_swap_lut_vec,n);
      common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);
      
      dim3 dec_row_col_swap_grid(m,n,1);
      dim3 dec_row_col_swap_blocks(channels,1,1);
      
      //Allocating GPU memory
      cudaMalloc((void**)&gpu_enc_vec,total * channels * sizeof(uint8_t));
      cudaMalloc((void**)&gpu_dec_vec,total * channels * sizeof(uint8_t));
      cudaMalloc((void**)&gpu_row_swap_lut_vec,m * sizeof(uint32_t));
      cudaMalloc((void**)&gpu_col_swap_lut_vec,n * sizeof(uint32_t));
      
      
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
    
    //Freeing GPU memory
    cudaFree((void*)gpu_enc_vec);
    cudaFree((void*)gpu_dec_vec);
    cudaFree((void*)gpu_row_swap_lut_vec);
    cudaFree((void*)gpu_col_swap_lut_vec); 
   }
    

  } 
  
  if(ROW_COL_ROTATION == 1)
  {
    /*Row and Column Unrotation*/
    cout<<"\nIn row and column unrotation ";
    for(int i = number_of_rounds - 1; i >= 0; --i)
    {
      
      cout<<"\nROUND "<<i;
      
      /*Vector generation for row and column unrotation*/
      x[0] = lm_parameters[i].x_init;
      y[0] = lm_parameters[i].y_init;
      pattern::twodLogisticMap(x,y,row_rotation_vec,lm_parameters[i].r,total * channels);
      
      x[0] = lma_parameters[i].x_init;
      y[0] = lma_parameters[i].y_init;
      pattern::twodLogisticMapAdvanced(x,y,col_rotation_vec,lma_parameters[i].myu1,lma_parameters[i].myu2,lma_parameters[i].lambda1,lma_parameters[i].lambda2,total * channels);
      
      common::genLUTVec(U,m);
      common::genLUTVec(V,n);
      common::rowColLUTGen(U,row_rotation_vec,V,col_rotation_vec,m,n);

      dim3 dec_gen_cat_map_grid(m,n,1);
      dim3 dec_gen_cat_map_blocks(channels,1,1);
      //Allocating GPU memory
      cudaMalloc((void**)&gpu_dec_vec,total * channels * sizeof(uint8_t));
      cudaMalloc((void**)&gpu_final_vec,total * channels * sizeof(uint8_t));
      cudaMalloc((void**)&gpu_U,m * sizeof(uint32_t));
      cudaMalloc((void**)&gpu_V,n * sizeof(uint32_t));
      
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
      
      //Free GPU memory
      cudaFree((void*)gpu_dec_vec);
      cudaFree((void*)gpu_final_vec);
      cudaFree((void*)gpu_U);
      cudaFree((void*)gpu_V);
      
      //Swapping input and output vectors
      
      
    }  
  }
 
  return 0;
}
