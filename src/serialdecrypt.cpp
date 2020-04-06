#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"

int main()
{
  cv::Mat3b image;
  long double time_array[15];
  long double total_time = 0.00;  

  clock_t img_read_start = clock();
  image = cv::imread("airplane_encrypted.png",cv::IMREAD_COLOR);
  clock_t img_read_end = clock();
  time_array[0] = 1000.0 * (img_read_end - img_read_start) / CLOCKS_PER_SEC;
    
  if(!image.data)
  {
    cout<<"\nCould not load encrypted image from "<<config::encrypted_image_path<<"\n Exiting...";
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
  
  /*Vector Declarations*/
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
  
  clock_t img_declaration_start =  clock();   
  cv::Mat3b imgout(m,n);
  clock_t img_declaration_end = clock();
  time_array[2] = (img_declaration_end - img_declaration_start) / CLOCKS_PER_SEC;
  
  clock_t row_random_vec_start = clock();
  pattern::MTSequence(row_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t row_random_vec_end = clock();
  time_array[3] = 1000.0 * (row_random_vec_end - row_random_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_random_vec_start = clock();
  pattern::MTSequence(col_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t col_random_vec_end = clock();
  time_array[4] = 1000.0 * (col_random_vec_end - col_random_vec_start) / CLOCKS_PER_SEC;
  
  clock_t row_lut_start = clock();
  common::genLUTVec(row_swap_lut_vec,m);
  clock_t row_lut_end = clock();
  time_array[5] = 1000.0 * (row_lut_end - row_lut_start) / CLOCKS_PER_SEC;
  
  clock_t col_lut_start = clock();
  common::genLUTVec(col_swap_lut_vec,n);
  clock_t col_lut_end = clock();
  time_array[6] = 1000.0 * (col_lut_end - col_lut_start) / CLOCKS_PER_SEC;
  
  clock_t swap_lut_start = clock();
  common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);    
  clock_t swap_lut_end = clock();
  time_array[7] = 1000.0 * (swap_lut_end - swap_lut_start) / CLOCKS_PER_SEC;
  
  //Diffusion Phase
  
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
  serial::grayLevelTransform(image,random_array,total);  
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
      printf(" %d",row_swap_lut_vec[i]);
    }
    
    cout<<"\nColumn LUT vector after swap = ";
    for(int i = 0; i < n; ++i)
    {
      printf(" %d",col_swap_lut_vec[i]);
    }
  }  
   
    //Image Unpermutation Phase
    
    //Unpermutation
    clock_t rotate_start = clock();
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < m; j++)
        {
            serial::rowRotator(imgout, image.row(j), j, n-row_swap_lut_vec[j], n);
        }
        
        for (int j = 0; j < n; j++)
        {
            serial::columnRotator(image, imgout.col(j), j, m-col_swap_lut_vec[j], m);
        }
    }
    clock_t rotate_end = clock();
    time_array[10] = 1000.0 * (rotate_end - rotate_start) / CLOCKS_PER_SEC; 
   
   image = image.reshape(1,image.rows * image.cols);
   imgout = imgout.reshape(1,imgout.rows * imgout.cols);
   
  //Row and Column Swapping
  clock_t swap_start = clock();
  serial::rowColSwapDec(image,imgout,row_swap_lut_vec,col_swap_lut_vec,m,n,total);
  clock_t swap_end = clock();
  time_array[11] = 1000.0 * (swap_end - swap_start) / CLOCKS_PER_SEC;
  
  
  image = image.reshape(3,m);
  imgout = imgout.reshape(3,m);
  
  
  
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nImage after decryption = ";
    common::printImageContents(image);
    cout<<"\n\nimgout in serialdecrypt";
    common::printImageContents(imgout);
  }
  

  clock_t img_write_start = clock();
  cv::imwrite("airplane_decrypted.png",image);
  clock_t img_write_end = clock();
  time_array[12] = 1000.0 * (img_write_end - img_write_start) / CLOCKS_PER_SEC;
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
  
  
  return 0;
}

