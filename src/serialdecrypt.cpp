#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/diagnostics.hpp"

int main()
{
  cv::Mat3b image;
  long double time_array[20];
  long double total_time = 0.00;  
  
  for(int i = 0; i < 17; ++i)
  {
    time_array[i] = 0;
  }  

  clock_t img_read_start = clock();
  image = cv::imread(config::diffused_image_path,cv::IMREAD_COLOR);
  clock_t img_read_end = clock();
  time_array[0] = 1000.0 * (img_read_end - img_read_start) / CLOCKS_PER_SEC;
    
  if(!image.data)
  {
    cout<<"\nCould not load encrypted image from "<<config::diffused_image_path<<"\n Exiting...";
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
  
  uint16_t *random_array = (uint16_t*)calloc(total * 3,sizeof(uint16_t));
  clock_t arr_declare_end = clock();
  time_array[3] = 1000.0 * (arr_declare_end - arr_declare_start) / CLOCKS_PER_SEC;
  
  clock_t img_declaration_start =  clock();   
  cv::Mat3b imgout(m,n);
  clock_t img_declaration_end = clock();
  time_array[2] = (img_declaration_end - img_declaration_start) / CLOCKS_PER_SEC;
  
  clock_t row_random_vec_start = clock();
  pattern::MTSequence(row_random_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  clock_t row_random_vec_end = clock();
  time_array[3] = 1000.0 * (row_random_vec_end - row_random_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_random_vec_start = clock();
  pattern::MTSequence(col_random_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_lut_gen);
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
 
  clock_t row_rotation_vec_start = clock();
  pattern::MTSequence(row_rotation_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_row_rotate);
  clock_t row_rotation_vec_end = clock();
  time_array[15] = 1000.0 * (row_rotation_vec_end - row_rotation_vec_start) / CLOCKS_PER_SEC;
  
  clock_t col_rotation_vec_start = clock();
  pattern::MTSequence(col_rotation_vec,total * 3,config::lower_limit,config::upper_limit,config::seed_col_rotate);
  clock_t col_rotation_vec_end = clock();
  time_array[16] = 1000.0 * (col_rotation_vec_end - col_rotation_vec_start) / CLOCKS_PER_SEC;
  
  if(DEBUG_VECTORS == 1)
  {
      cout<<"\nRow random vector = ";
      common::printArray16(row_random_vec, total * 3);
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
  
  //Diffusion Phase
  
  //Gray Level Transform
  if(DIFFUSION == 1)
  {

    config::slmm_map.x_init = 0.1;
    config::slmm_map.y_init = 0.1;
    config::slmm_map.alpha = 1.00;
    config::slmm_map.beta = 3.00;
    x[0] = config::slmm_map.x_init;
    y[0] = config::slmm_map.y_init;
  
    clock_t chaotic_map_start = clock();
    pattern::twodSineLogisticModulationMap(x,y,random_array,config::slmm_map.alpha,config::slmm_map.beta,total * 3);
    clock_t chaotic_map_end = clock();
    time_array[8] = 1000.0 * (chaotic_map_end - chaotic_map_start) / CLOCKS_PER_SEC;
  
    clock_t gray_level_transform_start = clock();
    serial::grayLevelTransform(image,random_array,total);  
    clock_t gray_level_transform_end = clock();
    time_array[9] = (gray_level_transform_end - gray_level_transform_start) / CLOCKS_PER_SEC;
  
    
    
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nAfter diffusion image = ";
      common::printImageContents(image);
      //cout<<"\n\nimgout = ";
      //common::printImageContents(imgout);
    }    

    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      clock_t write_undiffused_image_start = clock();
      bool write_undiffused_image = cv::imwrite(config::undiffused_image_path,image);
      clock_t write_undiffused_image_end = clock();
      time_array[13] = 1000.0 * (write_undiffused_image_end - write_undiffused_image_start) / CLOCKS_PER_SEC; 
      cout<<"\nWrite undiffused image = "<<write_undiffused_image;
    }
   }  
    

     

  //Row and Column Unswapping
  if(ROW_COL_SWAP == 1)
  {
    clock_t swap_start = clock();
    serial::rowColSwapDec(image,imgout,row_swap_lut_vec,col_swap_lut_vec,m,n,total);
    clock_t swap_end = clock();
    time_array[11] = 1000.0 * (swap_end - swap_start) / CLOCKS_PER_SEC;
    
  
    if(PRINT_IMAGES == 1)
    {
      cout<<"\nAfter row and column unswap image = ";
      common::printImageContents(image);
      cout<<"\n\nimgout = ";
      common::printImageContents(imgout);
    }
    
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      clock_t write_unswapped_image_start = clock();
      bool write_unswapped_image = cv::imwrite(config::row_col_unswapped_image_path,imgout);
      clock_t write_unswapped_image_end = clock();
      time_array[14] = 1000.0 * (write_unswapped_image_end - write_unswapped_image_start) / CLOCKS_PER_SEC;
      cout<<"\nWrite unswapped image = "<<write_unswapped_image;
    }
   }
  //Image Unpermutation Phase
    
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

      //Unpermutation
      clock_t rotate_start = clock();
      for (int i = 0; i < 1; i++)
      {
          for (int j = 0; j < m; j++)
          {
              serial::rowRotator(image, imgout.row(j), j, n-U[j], n);
          }
        
          for (int j = 0; j < n; j++)
          {
              serial::columnRotator(imgout, image.col(j), j, m-V[j], m);
          }
      }
      
      clock_t rotate_end = clock();
      time_array[10] = 1000.0 * (rotate_end - rotate_start) / CLOCKS_PER_SEC;
      
      if(PRINT_IMAGES == 1)
      {
        cout<<"\nAfter row and column unrotation image = ";
        common::printImageContents(image);
        cout<<"\n\nimgout = ";
        common::printImageContents(imgout);
      }      
      
      

      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
         clock_t img_write_start = clock();
         bool write_unrotated_image = cv::imwrite(config::row_col_unrotated_image_path,imgout);
         clock_t img_write_end = clock();
         time_array[12] = 1000.0 * (img_write_end - img_write_start) / CLOCKS_PER_SEC;
         cout<<"\nWrite unrotated image = "<<write_unrotated_image; 
      } 
      cv::Mat image_plain = cv::imread("output/airplane_1024_1024.png",IMREAD_COLOR);
      diagnostics::checkImageMatrices(imgout,image_plain);
}
  
  

  if(PRINT_TIMING == 1)
  {
    for(int i = 0; i < 17; ++i)
    {
      total_time += time_array[i];
    }
    
    total_time = total_time + config::img_path_init_time;
    
    printf("\nImage path initialization = %Lf ms",config::img_path_init_time);
    printf("\nRead image = %LF ms",time_array[0]);
    printf("\nVector declarations = %LF ms",time_array[1]);
    printf("\nImage declaration = %LF ms",time_array[2]);
    printf("\nRow random vector generation = %LF ms",time_array[3]);
    printf("\nColumn random vector generation = %LF ms",time_array[4]);
    printf("\nRow LUT vector generation = %LF ms",time_array[5]);
    printf("\nColumn LUT vector generation = %LF ms",time_array[6]);
    printf("\nRow rotation vector generation = %Lf ms",time_array[15]);
    printf("\nColumn rotation vector generation = %Lf ms",time_array[16]);
    printf("\nSwap LUT = %LF ms",time_array[7]);
    printf("\nChaotic Map = %LF ms",time_array[8]);
    printf("\nGray level transform = %LF ms",time_array[9]);
    printf("\nRow and column unswap = %LF ms",time_array[11]);
    printf("\nImage unrotation = %LF ms",time_array[10]);
    printf("\nWrite undiffused image = %LF ms",time_array[13]);
    printf("\nWrite row and column unrotated image = %Lf ms",time_array[12]);
    printf("\nWrite row and column unswapped image = %Lf ms",time_array[14]);
    printf("\nTotal time = %LF ms",total_time);  
  
  }
    
  return 0;
}

