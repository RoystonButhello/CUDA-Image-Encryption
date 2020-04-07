#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"


int main()
{
  cv::Mat3b image;
  long double time_array[15];
  long double total_time = 0.0000;  

  clock_t img_read_start = clock();
  image = cv::imread(config::diffused_image_path,cv::IMREAD_COLOR);
  clock_t img_read_end = clock();
  time_array[0] = 1000.0 * (img_read_end - img_read_start) / CLOCKS_PER_SEC;
    
  if(!image.data)
  {
    cout<<"\nCould not load encrypted image "<<"airplane_encrypted.png "<<"\n Exiting...";
    exit(0);
  }
  
  if(RESIZE_TO_DEBUG == 1)
  {
    cv::resize(image,image,cv::Size(config::cols,config::rows),CV_INTER_LANCZOS4);
  }  
  
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nEncrypted image = ";
    common::printImageContents(image);
  }  

  uint32_t m = 0,n = 0,channels = 0,total = 0;

  
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  
  cout<<"\nRows = "<<m;
  cout<<"\nColumns = "<<n;
  cout<<"\nChannels = "<<channels;
  
  /*CPU Declarations*/
  
  clock_t vec_declaration_start = clock();
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *row_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *col_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * n);
  double *x = (double*)malloc(sizeof(double) * total * 3);
  double *y = (double*)malloc(sizeof(double) * total * 3);
  uint32_t *random_array = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  clock_t vec_declaration_end = clock();
  time_array[1] = 1000.0 * (vec_declaration_end - vec_declaration_start) / CLOCKS_PER_SEC;
  
  /*GPU Declarations*/
  uint8_t *gpu_enc_vec;
  uint8_t *gpu_dec_vec;
  uint32_t *gpu_row_swap_lut_vec;
  uint32_t *gpu_col_swap_lut_vec;
  cudaMallocManaged((void**)&gpu_enc_vec,total * 3 * sizeof(uint8_t));
  cudaMallocManaged((void**)&gpu_dec_vec,total * 3 * sizeof(uint8_t));
  cudaMallocManaged((void**)&gpu_row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMallocManaged((void**)&gpu_col_swap_lut_vec,n * sizeof(uint32_t));
  
  //Flattening Image
  common::flattenImage(image,gpu_enc_vec);
  
  clock_t row_random_vec_start = clock();
  pattern::MTSequence(row_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t row_random_vec_end = clock();
  time_array[3] = 1000.0 * (row_random_vec_end - row_random_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_random_vec_start = clock();
  pattern::MTSequence(col_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t col_random_vec_end = clock();
  time_array[4] = 1000.0 * (col_random_vec_end - col_random_vec_start) / CLOCKS_PER_SEC;
  
  clock_t row_lut_start = clock();
  common::genLUTVec(gpu_row_swap_lut_vec,m);
  clock_t row_lut_end = clock();
  time_array[5] = 1000.0 * (row_lut_end - row_lut_start) / CLOCKS_PER_SEC;
  
  clock_t col_lut_start = clock();
  common::genLUTVec(gpu_col_swap_lut_vec,n);
  clock_t col_lut_end = clock();
  time_array[6] = 1000.0 * (col_lut_end - col_lut_start) / CLOCKS_PER_SEC;
  
  clock_t swap_lut_start = clock();
  common::rowColLUTGen(gpu_row_swap_lut_vec,row_random_vec,gpu_col_swap_lut_vec,col_random_vec,m,n);    
  clock_t swap_lut_end = clock();
  time_array[7] = 1000.0 * (swap_lut_end - swap_lut_start) / CLOCKS_PER_SEC;
  
  //Diffusion Phase
  
  if(DIFFUSION == 1)
  {
    //Gray Level Transform
  
    config::slmm_map.x_init = 0.1;
    config::slmm_map.y_init = 0.1;
    config::slmm_map.alpha = 1.00;
    config::slmm_map.beta = 3.00;
    x[0] = config::slmm_map.x_init;
    y[0] = config::slmm_map.y_init;
  
    clock_t chaotic_map_start = clock();
    pattern::twodSineLogisticModulationMap(x,y,random_array,config::slmm_map.alpha,config::slmm_map.beta,total);
    clock_t chaotic_map_end = clock();
    time_array[8] = 1000.0 * (chaotic_map_end - chaotic_map_start) / CLOCKS_PER_SEC;
  
    clock_t gray_level_transform_start = clock();
    serial::grayLevelTransform(gpu_enc_vec,random_array,total);  
    clock_t gray_level_transform_end = clock();
    time_array[9] = (gray_level_transform_end - gray_level_transform_start) / CLOCKS_PER_SEC;
  
    if(DEBUG_VECTORS == 1)
    {
    
      cout<<"\n\nRow random vector = ";
      for(int i = 0; i < total * 3; ++i)
      {
        printf(" %d",row_random_vec[i]);
      }
    
      cout<<"\n\nColumn random vector = ";
      for(int i = 0; i < total * 3; ++i)
      {
        printf(" %d",col_random_vec[i]);
      }
    
    } 
  
    if(DEBUG_VECTORS == 1)
    {
      cout<<"\nRow LUT vector after swap = ";
      for(int i = 0; i < m; ++i)
      {
        printf(" %d",gpu_row_swap_lut_vec[i]);
      }
    
      cout<<"\nColumn LUT vector after swap = ";
      for(int i = 0; i < n; ++i)
      {
        printf(" %d",gpu_col_swap_lut_vec[i]);
      }
    }  
    
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      if(DEBUG_VECTORS == 1)
      {
         cout<<"\nUndiffused image = ";
         for(int i = 0; i < total * 3; ++i)
         {
           printf(" %d",gpu_enc_vec[i]);
         }  
      }
      
      cv::Mat img_reshape(m,n,CV_8UC3,gpu_enc_vec);
      cv::imwrite(config::undiffused_image_path,img_reshape);
    }   
    
   }
    //Image Unpermutation Phase
    
    //Unpermutation
    if(ROW_COL_ROTATION == 1)
    {
      
      dim3 grid_dec_gen_cat_map(m,n,1);
      dim3 block_dec_gen_cat_map(3,1,1);
      run_DecGenCatMap(gpu_enc_vec,gpu_dec_vec,gpu_col_swap_lut_vec,gpu_row_swap_lut_vec,grid_dec_gen_cat_map,block_dec_gen_cat_map);
      
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        if(DEBUG_VECTORS == 1)
        {
          cout<<"\nUnpermuted image = ";
          for(int i = 0; i < total * 3; ++i)
          {
            printf(" %d",gpu_dec_vec[i]);
          }
        }
        cv::Mat img_reshape(m,n,CV_8UC3,gpu_dec_vec);
        cv::imwrite(config::row_col_unpermuted_image_path,img_reshape);
        
      }
    }
   
 
  
  if(PRINT_TIMING == 1)
  {
    
    for(int i = 0; i < 13; ++i)
    {
      total_time += time_array[i];
    }
    
    printf("\nRead image = %LF ms",time_array[0]);
    printf("\nVector declarations = %LF ms",time_array[1]);
    printf("\nImage declaration = %LF ms",time_array[2]);
    printf("\nRow random vector generation = %LF ms",time_array[3]);
    printf("\nColumn random vector generation = %LF ms",time_array[4]);
    printf("\nRow LUT vector generation = %LF ms",time_array[5]);
    printf("\nColumn LUT vector generation = %LF ms",time_array[6]);
    printf("\nSwap LUT = %LF ms",time_array[7]);
    printf("\nChaotic Map = %LF ms",time_array[8]);
    printf("\nGray level transform = %LF ms",time_array[9]);
    printf("\nImage unrotation = %LF ms",time_array[10]);
    printf("\nRow and column unswap = %LF ms",time_array[11]);
    printf("\nWrite image = %LF ms",time_array[12]);
    printf("\nTotal time = %LF ms",total_time);  
  }
  
  return 0;
}


