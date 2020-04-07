#include "include/commonheader.hpp"
#include "include/pattern.hpp"
#include "include/serial.hpp"
#include "include/kernel.hpp"

int main()
{
  cv::Mat3b image;
  long double time_array[16];
  long double total_time = 0.0000;
  clock_t img_read_start = clock();
  image = cv::imread(config::input_image_path,cv::IMREAD_COLOR);
  clock_t img_read_end = clock();
  time_array[0] = 1000.0 * (img_read_end - img_read_start) / CLOCKS_PER_SEC;
 
  if(!image.data)
  {
    cout<<"\nCould not load input image from "<<config::input_image_path<<"\n Exiting...";
    exit(0);
  }
  
  if(RESIZE_TO_DEBUG == 1)
  {
    clock_t img_resize_start = clock();
    cv::resize(image,image,cv::Size(config::cols,config::rows),CV_INTER_LANCZOS4);
    clock_t img_resize_end = clock();
    //time_array[1] = 1000.0 * (img_resize_end - img_resize_start) / CLOCKS_PER_SEC;
  }  
  
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nOriginal image = ";
    common::printImageContents(image);
  }  

  uint32_t m = 0,n = 0,channels = 0,total = 0;
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  
  cout<<"\nm = "<<m;
  cout<<"\nn = "<<n;
  cout<<"\nChannels = "<<channels;
  
  

  /*CPU Declarations*/
  Mat3b imgout(m,n);
  clock_t img_declare_start = clock();
  clock_t img_declare_end = clock();
  time_array[2] = 1000.0 * (img_declare_end - img_declare_start) / CLOCKS_PER_SEC;
  clock_t arr_declare_start = clock();
  
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  double *x = (double*)malloc(sizeof(double) * total * 3);
  double *y = (double*)malloc(sizeof(double) * total * 3);
  uint32_t *random_array = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  //uint32_t *row_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * m);
  //uint32_t *col_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * n);
  clock_t arr_declare_end = clock();
  time_array[3] = 1000.0 * (arr_declare_end - arr_declare_start) / CLOCKS_PER_SEC;
  
  /*GPU Declarations*/
  uint8_t *gpu_img_vec;
  uint8_t *gpu_enc_vec;
  uint32_t *gpu_row_swap_lut_vec;
  uint32_t *gpu_col_swap_lut_vec;
  cudaMallocManaged((void**)&gpu_img_vec,total * 3 * sizeof(uint8_t));
  cudaMallocManaged((void**)&gpu_enc_vec,total * 3 * sizeof(uint8_t));
  cudaMallocManaged((void**)&gpu_row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMallocManaged((void**)&gpu_col_swap_lut_vec,n * sizeof(uint32_t));
    

  /*Random Vector Generation*/
  clock_t row_vec_start = clock();
  pattern::MTSequence(row_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t row_vec_end = clock();
  time_array[4] = 1000.0 * (row_vec_end - row_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_vec_start = clock();
  pattern::MTSequence(col_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t col_vec_end = clock();
  time_array[5] = 1000.0 * (col_vec_end - col_vec_start) / CLOCKS_PER_SEC;
  
  clock_t row_lut_vec_start = clock();
  common::genLUTVec(gpu_row_swap_lut_vec,m);
  clock_t row_lut_vec_end = clock();
  time_array[6] = 1000.0 * (row_lut_vec_end - row_lut_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_lut_vec_start = clock();
  common::genLUTVec(gpu_col_swap_lut_vec,n);
  clock_t col_lut_vec_end = clock();
  time_array[7] = 1000.0 * (col_lut_vec_end - col_lut_vec_start) / CLOCKS_PER_SEC;
  
  clock_t lut_vec_swap_start = clock();
  common::rowColLUTGen(gpu_row_swap_lut_vec,row_random_vec,gpu_col_swap_lut_vec,col_random_vec,m,n);    
  clock_t lut_vec_swap_end = clock();
  time_array[8] = 1000.0 * (lut_vec_swap_end - lut_vec_swap_start) / CLOCKS_PER_SEC;
  
  
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
   
  
   
  /*Image Permutation Phase*/
  
  /*Row and column Rotation*/
  
  if(ROW_COL_ROTATION == 1)
  {
    common::flattenImage(image,gpu_img_vec);
    dim3 grid_enc_gen_cat_map(m,n,1);
    dim3 block_enc_gen_cat_map(3,1,1);
    run_EncGenCatMap(gpu_img_vec,gpu_enc_vec,gpu_col_swap_lut_vec,gpu_row_swap_lut_vec,grid_enc_gen_cat_map,block_enc_gen_cat_map);
    if(DEBUG_VECTORS == 1)
    {
      cout<<"\ngpu_img_vec = ";
      for(int i = 0; i < total * 3; ++i)
      {
        printf("%d ",gpu_img_vec[i]);
        
      }
      cout<<"\ngpu_enc_vec = ";
      for(int i = 0; i < total * 3; ++i)
      {
        printf(" %d",gpu_enc_vec[i]);
      }
    }    

    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      
      cv::Mat img_reshape(m,n,CV_8UC3,gpu_enc_vec);
      cv::imwrite(config::row_col_permuted_image_path,img_reshape);
      
      if(PRINT_IMAGES == 1)
      {
        cout<<"\nImage after permutation = ";
        common::printImageContents(img_reshape);
      }
      
    }
    
    
  }
  
  
  
  
  /*Diffusion Phase*/
  
  if(DIFFUSION == 1)
  { 
  
    /*Gray Level Transform*/
    config::slmm_map.x_init = 0.1;
    config::slmm_map.y_init = 0.1;
    config::slmm_map.alpha = 1.00;
    config::slmm_map.beta = 3.00;
   
  
    x[0] = config::slmm_map.x_init;
    y[0] = config::slmm_map.y_init;
    
  
    clock_t chaotic_map_start = clock();
    pattern::twodSineLogisticModulationMap(x,y,random_array,config::slmm_map.alpha,config::slmm_map.beta,total);
    clock_t chaotic_map_end = clock();
    time_array[12] = 1000.0 * (chaotic_map_end - chaotic_map_start) / CLOCKS_PER_SEC;
  
    clock_t gray_level_transform_start = clock();
    serial::grayLevelTransform(gpu_enc_vec,random_array,total);
    clock_t gray_level_transform_end = clock();
    time_array[13] = 1000.0 * (gray_level_transform_end - gray_level_transform_start) / CLOCKS_PER_SEC;
    
    if(DEBUG_VECTORS == 1)
    {
      cout<<"\nDiffused image = ";
      for(int i = 0; i < total * 3; ++i)
      {
       
        printf(" %d",gpu_enc_vec[i]);
      }
    }   
  
    cv::Mat img_reshape(m,n,CV_8UC3,gpu_enc_vec);
    
    clock_t img_write_start = clock();
    cv::imwrite(config::diffused_image_path,img_reshape);
    clock_t img_write_end = clock();
    time_array[14] = 1000.0 * (img_write_end - img_write_start) / CLOCKS_PER_SEC;
     
  } 
  
 

    

  if(PRINT_TIMING == 1)
  {
    
    for(int i = 0; i < 16; ++i)
    {
      if(i != 1)
      {
        total_time += time_array[i];
      }
    }
    
    printf("\nRead image = %Lf ms",time_array[0]);
    printf("\nEmpty image declaration = %Lf ms",time_array[2]);
    printf("\nVector declarations = %Lf ms",time_array[3]);
    printf("\nGenerate row random vector = %Lf ms",time_array[4]);
    printf("\nGenerate column random vector = %Lf ms",time_array[5]);
    printf("\nGenerate row LUT vector = %Lf ms",time_array[6]);
    printf("\nGenerate column LUT vector = %Lf ms",time_array[7]);
    printf("\nSwap LUT vectors = %Lf ms",time_array[8]);
    printf("\nImage permutation = %Lf ms",time_array[9]);
    printf("\nImage reshape = %Lf ms",time_array[10]);
    printf("\nRow and column swap = %Lf ms",time_array[11]);
    printf("\nChaotic map = %Lf ms",time_array[12]);
    printf("\nGray level transform = %Lf ms",time_array[13]);
    printf("\nWrite encrypted image = %Lf ms",time_array[14]);
    printf("\nTotal time = %Lf ms",total_time); 
  }
    
  return 0;
}


