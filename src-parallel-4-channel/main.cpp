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
  int number_of_rounds = 4;
  int fseek_status = 9,fwrite_status = 9;
  
  long ptr_position = 0;
  double alpha = 0.00,beta = 0.00;
  cudaEvent_t start_rotate,stop_rotate,start_swap,stop_swap;


  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  //cv::Mat4b img_dec(m,n);
  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
  
  /*Parameter arrays*/
  config::lm lm_parameters[number_of_rounds];
  config::lma lma_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::lalm lalm_parameters[number_of_rounds];
  config::mt mt_parameters[0];
  
  /*Initializing parameters to zero for each round*/
  common::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
  /*Assigning parameters for each round*/
  common::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
  /*Writing parameters*/
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
  
  ptr_position = writeLMParameters(outfile,config::constant_parameters_file_path,"ab",lm_parameters,0,number_of_rounds,ptr_position);
  ptr_position = writeLMAParameters(outfile,config::constant_parameters_file_path,"ab",lma_parameters,0,number_of_rounds,ptr_position);
  ptr_position = writeSLMMParameters(outfile,config::constant_parameters_file_path,"ab",slmm_parameters,0,number_of_rounds,ptr_position);
  ptr_position = writeLASMParameters(outfile,config::constant_parameters_file_path,"ab",lasm_parameters,0,number_of_rounds,ptr_position);
  ptr_position = writeMTParameters(outfile,config::constant_parameters_file_path,"ab",mt_parameters,0,1,ptr_position);
  
  /*Display parameters after writing*/
  common::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
  /*CPU vector declarations*/
  uint8_t *img_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * channels);
  uint8_t *enc_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * channels);
  uint8_t *final_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * channels);
  uint32_t *row_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *col_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * n);
  uint32_t *U = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *V = (uint32_t*)malloc(sizeof(uint32_t) * n);
  double *x = (double*)malloc(sizeof(double) * total * channels);
  double *y = (double*)malloc(sizeof(double) * total * channels);
  uint32_t *row_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *col_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *diffusion_array = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  
  /*GPU vector declarations*/
  uint8_t *gpu_img_vec;
  uint8_t *gpu_enc_vec;
  uint8_t *gpu_final_vec;
  uint32_t *gpu_U;
  uint32_t *gpu_V;
  uint32_t *gpu_row_swap_lut_vec;
  uint32_t *gpu_col_swap_lut_vec;
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
  
  
  if(ROW_COL_ROTATION == 1)
  {
    cout<<"\nIn row and column rotation";
    for(int i = 0; i < number_of_rounds; ++i)
    {
      cout<<"\nROUND "<<i;
      
      /*Vector generation for row and column rotation*/
      x[0] = lm_parameters[i].x_init;
      y[0] = lm_parameters[i].y_init;
      pattern::twodLogisticMap(x,y,row_rotation_vec,lm_parameters[i].r,total * channels);
      
      x[0] = lma_parameters[i].x_init;
      y[0] = lma_parameters[i].y_init;
      pattern::twodLogisticMapAdvanced(x,y,col_rotation_vec,lma_parameters[i].myu1,lma_parameters[i].myu2,lma_parameters[i].lambda1,lma_parameters[i].lambda2,total * channels);
      
      common::genLUTVec(U,m);
      common::genLUTVec(V,n);
      common::rowColLUTGen(U,row_rotation_vec,V,col_rotation_vec,m,n);  
      
      /*Row and Column Rotation*/

      dim3 enc_gen_cat_map_grid(m,n,1);
      dim3 enc_gen_cat_map_blocks(channels,1,1);
      
      //Copying CPU vectors to GPU memory
      cudaMemcpy(gpu_img_vec,img_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_U,U,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_V,V,n * sizeof(uint32_t),cudaMemcpyHostToDevice);
        
      run_EncGenCatMap(gpu_img_vec,gpu_enc_vec,gpu_V,gpu_U,enc_gen_cat_map_grid,enc_gen_cat_map_blocks);
      
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(img_vec,gpu_img_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      if(number_of_rounds > 1)
      {  
        cudaMemcpy(img_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToHost);
        cudaMemcpy(gpu_img_vec,img_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      }
      
      //std::memcpy(img_vec,enc_vec,total * channels);
      
      
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        config::rotated_image = "";
        config::rotated_image = config::image_name + "_rotated" + "_ROUND_" + std::to_string(i) + config::extension;
        
        cout<<"\n"<<config::rotated_image;
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
    for(int i = 0; i < number_of_rounds; ++i)
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
      
      
      dim3 enc_row_col_swap_grid(m,n,1);
      dim3 enc_row_col_swap_blocks(channels,1,1);
      
      //Transferring vectors from CPU to GPU memory
      cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_final_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_row_swap_lut_vec,row_swap_lut_vec,m * sizeof(uint32_t),cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_col_swap_lut_vec,col_swap_lut_vec,n * sizeof(uint32_t),cudaMemcpyHostToDevice);      

      run_encRowColSwap(gpu_enc_vec,gpu_final_vec,gpu_row_swap_lut_vec,gpu_col_swap_lut_vec,enc_row_col_swap_grid,enc_row_col_swap_blocks);
      
      cudaMemcpy(final_vec,gpu_final_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      cudaMemcpy(enc_vec,gpu_enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyDeviceToHost);
      
      if(number_of_rounds  > 1)
      {
        cudaMemcpy(enc_vec,final_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToHost);
        cudaMemcpy(gpu_enc_vec,enc_vec,total * channels * sizeof(uint8_t),cudaMemcpyHostToDevice);
      }
      //std::memcpy(enc_vec,final_vec,total * channels);
      
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
    
    pattern::MTSequence(diffusion_array,total * channels,config::lower_limit,config::upper_limit,mt_parameters[0].seed_1);
    serial::grayLevelTransform(final_vec,diffusion_array,total * channels);

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
