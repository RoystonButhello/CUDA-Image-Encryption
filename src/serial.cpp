#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"


int main()
{
  cv::Mat3b image;
  long double time_array[21];
  long double total_time = 0.0000;
  size_t operating_rounds = 2;
  
  
  /*Initialize time array*/
  for(int i = 0; i < 20; ++i)
  {
   time_array[i] = 0;
  }
  
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
    cv::imwrite("output/airplane_1024_1024.png",image);
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


  /*Vector and image Declarations*/
  clock_t img_declare_start = clock();
  Mat3b imgout(m,n); 
  clock_t img_declare_end = clock();
  time_array[2] = 1000.0 * (img_declare_end - img_declare_start) / CLOCKS_PER_SEC;
  clock_t arr_declare_start = clock();
  
  uint16_t *row_random_vec = (uint16_t*)calloc(total * 3,sizeof(uint16_t));
  uint16_t *col_random_vec = (uint16_t*)calloc(total * 3,sizeof(uint16_t));
  
  uint16_t *row_swap_lut_vec = (uint16_t*)calloc(m,sizeof(uint16_t));
  uint16_t *col_swap_lut_vec = (uint16_t*)calloc(n,sizeof(uint16_t));
  
  uint16_t *row_rotation_vec = (uint16_t*)calloc(total * 3,sizeof(uint16_t));
  uint16_t *col_rotation_vec = (uint16_t*)calloc(total * 3,sizeof(uint16_t));
  
  uint16_t *U = (uint16_t*)calloc(m,sizeof(uint16_t));
  uint16_t *V = (uint16_t*)calloc(n,sizeof(uint16_t));
  
  double *x = (double*)calloc(total * 3,sizeof(double));
  double *y = (double*)calloc(total * 3,sizeof(double));
  
  uint16_t *random_array = (uint16_t*)calloc(total * 3,sizeof(uint32_t));
  clock_t arr_declare_end = clock();
  time_array[3] = 1000.0 * (arr_declare_end - arr_declare_start) / CLOCKS_PER_SEC;
  
  clock_t row_vec_start = clock();
  pattern::MTSequence(row_random_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t row_vec_end = clock();
  time_array[4] = 1000.0 * (row_vec_end - row_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_vec_start = clock();
  pattern::MTSequence(col_random_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t col_vec_end = clock();
  time_array[5] = 1000.0 * (col_vec_end - col_vec_start) / CLOCKS_PER_SEC;
  
  clock_t row_lut_vec_start = clock();
  common::genLUTVec(row_swap_lut_vec,m);
  clock_t row_lut_vec_end = clock();
  time_array[6] = 1000.0 * (row_lut_vec_end - row_lut_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_lut_vec_start = clock();
  common::genLUTVec(col_swap_lut_vec,n);
  clock_t col_lut_vec_end = clock();
  time_array[7] = 1000.0 * (col_lut_vec_end - col_lut_vec_start) / CLOCKS_PER_SEC;
  
  clock_t lut_vec_swap_start = clock();
  common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);    
  clock_t lut_vec_swap_end = clock();
  time_array[8] = 1000.0 * (lut_vec_swap_end - lut_vec_swap_start) / CLOCKS_PER_SEC;
  
  clock_t row_random_vec_start = clock();
  pattern::MTSequence(row_rotation_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_row_rotate);
  clock_t row_random_vec_end = clock();
  time_array[16] = 1000.0 * (row_random_vec_end - row_random_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_random_vec_start = clock();
  pattern::MTSequence(col_rotation_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_col_rotate);
  clock_t col_random_vec_end = clock();
  time_array[17] = 1000.0 * (col_random_vec_end - col_random_vec_start) / CLOCKS_PER_SEC;
  
  
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\n\nRow random vector = ";
    common::printArray16(row_random_vec,total * 3);
    cout<<"\n\nColumn random vector = ";
    common::printArray16(col_random_vec,total * 3);
    cout<<"\nRow LUT vector after swap = ";
    common::printArray16(row_swap_lut_vec,m);
    cout<<"\nColumn LUT vector after swap = ";
    common::printArray16(col_swap_lut_vec,n);
    cout<<"\nRow rotation vector = ";
    common::printArray16(row_rotation_vec,total * 3);
    cout<<"\nColumn rotation vector = ";
    common::printArray16(col_rotation_vec,total * 3);  
  } 
  

    
  /*Image Permutation Phase*/
  
  if(ROW_COL_ROTATE == 1)
  {
   common::genLUTVec(U,m);
   common::genLUTVec(V,n); 
   common::rowColLUTGen(U,row_rotation_vec,V,col_rotation_vec,m,n);   
   if(DEBUG_VECTORS == 1)
   {
     cout<<"\nU = ";
     common::printArray16(U,m);
     cout<<"\nV = ";
     common::printArray16(V,n);
   } 
    
    /*Row and column Rotation*/
    clock_t img_permute_start = clock();
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < n; j++) // For each column
        {
            serial::columnRotator(imgout, image.col(j), j, V[j], m);
        }
        
        for (int j = 0; j < m; j++) // For each row
        {
            serial::rowRotator(image, imgout.row(j), j, U[j], n);
        }
  
        clock_t img_permute_end = clock();
        time_array[9] = 1000.0 * (img_permute_end - img_permute_start) / CLOCKS_PER_SEC;
        
        if(PRINT_IMAGES == 1)
        {
          cout<<"\nAfter Row and Column rotation image = ";
          common::printImageContents(image);
          cout<<"\n\nimgout = ";
          common::printImageContents(imgout);
        }
        
        if(DEBUG_INTERMEDIATE_IMAGES == 1)
        {
            clock_t write_row_col_rotated_image_start = clock();
            bool write_rotated_image = cv::imwrite(config::row_col_rotated_image_path,image);
            clock_t write_row_col_rotated_image_end = clock();
            time_array[13] = 1000.0 * (write_row_col_rotated_image_end - write_row_col_rotated_image_start) / CLOCKS_PER_SEC;
            cout<<"\nWrite rotated image = "<<write_rotated_image;
        }
     }  
  }
  
  
   
  /*Row and Column Swapping*/
  if(ROW_COL_SWAP == 1)
  {
    

    clock_t row_col_swap_start = clock();
    serial::rowColSwapEnc(image,imgout,row_swap_lut_vec,col_swap_lut_vec,m,n,total);
    clock_t row_col_swap_end = clock();
    time_array[10] = 1000.0 * (row_col_swap_end - row_col_swap_start) / CLOCKS_PER_SEC;
    
    if(PRINT_IMAGES == 1)
    {
      
      
      cout<<"\nAfter row and column swapping image = ";
      common::printImageContents(image);
      cout<<"\n\nimgout = ";
      common::printImageContents(imgout);
    }    

    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      
      clock_t write_row_col_swapped_image_start = clock();
      bool write_swapped_image= cv::imwrite(config::row_col_swapped_image_path,imgout);
      clock_t write_row_col_swapped_image_end = clock();
      time_array[14] = 1000.0 * (write_row_col_swapped_image_end - write_row_col_swapped_image_start) / CLOCKS_PER_SEC;
      cout<<"\nWrite swapped image = "<<write_swapped_image;
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
     pattern::twodSineLogisticModulationMap(x,y,random_array,config::slmm_map.alpha,config::slmm_map.beta,total * 3);
     clock_t chaotic_map_end = clock();
     time_array[11] = 1000.0 * (chaotic_map_end - chaotic_map_start) / CLOCKS_PER_SEC;
    
     clock_t gray_level_transform_start = clock();
     serial::grayLevelTransform(imgout,random_array,total);
     clock_t gray_level_transform_end = clock();
     time_array[12] = 1000.0 * (gray_level_transform_end - gray_level_transform_start) / CLOCKS_PER_SEC;
     
     if(PRINT_IMAGES == 1)
     {
        cout<<"\nAfter diffusion ";
        //common::printImageContents(image);
        cout<<"\n\nimgout = ";
        common::printImageContents(imgout);
     }     
     
     if(DEBUG_VECTORS == 1)
     {
       cout<<"\nRandom Array = ";
       common::printArray16(random_array,total * 3);
     }
     
     if(DEBUG_INTERMEDIATE_IMAGES == 1)
     {
       clock_t write_diffused_image_start = clock();
       bool write_diffused_image = cv::imwrite(config::diffused_image_path,imgout);
       clock_t write_diffused_image_end = clock();
       time_array[15] = 1000.0 * (write_diffused_image_end - write_diffused_image_start) / CLOCKS_PER_SEC;
       cout<<"\nWrite diffused image = "<<write_diffused_image; 
     }
       
 }
 


  
  if(PRINT_TIMING == 1)
  {
    for(int i = 0; i < 18; ++i)
    {
      total_time += time_array[i];
    }
     
    total_time = total_time + config::img_path_init_time;

    printf("\nImage path initialization = %Lf ms",config::img_path_init_time);
    printf("\nRead image = %Lf ms",time_array[0]);
    printf("\nEmpty image declaration = %Lf ms",time_array[2]);
    printf("\nVector declarations = %Lf ms",time_array[3]);
    printf("\nGenerate row random vector = %Lf ms",time_array[4]);
    printf("\nGenerate column random vector = %Lf ms",time_array[5]);
    printf("\nGenerate row LUT vector = %Lf ms",time_array[6]);
    printf("\nGenerate column LUT vector = %Lf ms",time_array[7]);
    printf("\nSwap LUT vectors = %Lf ms",time_array[8]);
    printf("\nGenerate row rotation vector = %Lf ms",time_array[16]);
    printf("\nGenerate column rotation vector = %Lf ms",time_array[17]);
    printf("\nImage permutation = %Lf ms",time_array[9]);
    printf("\nRow and column swap = %Lf ms",time_array[10]);
    printf("\nChaotic map = %Lf ms",time_array[11]);
    printf("\nGray level transform = %Lf ms",time_array[12]);
    printf("\nWrite row and column rotated image = %Lf ms",time_array[13]);
    printf("\nWrite row and column swapped_image = %Lf ms",time_array[14]);
    printf("\nWrite diffused image = %Lf ms",time_array[15]);
    printf("\nTotal time = %Lf ms",total_time); 
  }
      
  return 0;
}

