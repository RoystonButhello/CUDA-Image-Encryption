#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"
static inline void vecDifference(uint8_t *img_plain,uint8_t *img_dec,uint32_t total);
static inline void imgDifference(cv::Mat plain_image,cv::Mat decrypted_image);

static inline void vecDifference(uint8_t *img_plain,uint8_t *img_dec,uint32_t total)
{ 
  int cnt = 0;
  uint8_t from = 0;
  
  
  for(int i = 0; i < total; ++i)
  {
    from = img_plain[i] - img_dec[i];
    if(from != 0)
    {
      ++cnt;
    }
    //printf("\n%d",img_plain[i] - img_dec[i]);
  }
  printf("\ncnt = %d",cnt);
}

static inline void imgDifference(cv::Mat plain_image,cv::Mat decrypted_image)
{
  int cnt = 0;
  uint8_t from = 0;
  for(int i = 0; i < plain_image.rows; ++i)
  {
    for(int j = 0; j < plain_image.cols; ++j)
    {
      for(int k = 0; k < plain_image.channels(); ++k)
      {
        from = plain_image.at<Vec3b>(i,j)[k] - decrypted_image.at<Vec3b>(i,j)[k];
        if(from == 0)
        {
          ++cnt;
        }
       
      }
    }
  }
  printf("\ncnt = %d",cnt);
}

int main()
{
  cv::Mat image;
  image = cv::imread(config::input_image_path,cv::IMREAD_UNCHANGED);
  
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
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  //cv::Mat4b img_dec(m,n);
  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
  uint8_t *img_vec;
  uint8_t *enc_vec;
  uint8_t *final_vec;
  uint32_t *row_swap_lut_vec;
  uint32_t *col_swap_lut_vec;
  uint32_t *U;
  uint32_t *V;
  cudaMallocManaged((void**)&img_vec,total * channels * sizeof(uint8_t));
  cudaMallocManaged((void**)&enc_vec,total * channels * sizeof(uint8_t));
  cudaMallocManaged((void**)&final_vec,total * channels * sizeof(uint8_t));
  cudaMallocManaged((void**)&row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMallocManaged((void**)&col_swap_lut_vec,n * sizeof(uint32_t));
  cudaMallocManaged((void**)&U,m * sizeof(uint32_t));
  cudaMallocManaged((void**)&V,n * sizeof(uint32_t));
  
  uint32_t *row_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *col_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  
  /*Vector generation for row and column rotation*/
  common::genLUTVec(U,m);
  common::genLUTVec(V,n);
  pattern::MTSequence(row_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_row_rotate);
  pattern::MTSequence(col_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_col_rotate);
  common::rowColLUTGen(U,row_rotation_vec,V,col_rotation_vec,m,n);   

  /*Vector generation for row and column swapping*/
  common::genLUTVec(row_swap_lut_vec,m);
  common::genLUTVec(col_swap_lut_vec,n);
  pattern::MTSequence(row_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_lut_gen_1);
  pattern::MTSequence(col_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_lut_gen_2);
  common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);  
  

  
  /*Flattening image*/
  common::flattenImage(image,img_vec,channels);
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\n\nInput image = ";
    common::printArray8(img_vec,total * channels);
    /*cout<<"\nRow random vec = ";
    common::printArray32(row_random_vec,total * channels);
    cout<<"\nColumn random vec = ";
    common::printArray32(col_random_vec,total * channels);
    cout<<"\nRow Swap LUT vec = ";
    common::printArray32(row_swap_lut_vec,m);
    cout<<"\nCol Swap LUT vec = ";
    common::printArray32(col_swap_lut_vec,n);*/
  }
  
  if(ROW_COL_ROTATION == 1)
  {
    /*Row and Column Rotation*/
    cout<<"\nIn row and column rotation";
    dim3 enc_gen_cat_map_grid(m,n,1);
    dim3 enc_gen_cat_map_blocks(channels,1,1);
    run_EncGenCatMap(img_vec,enc_vec,V,U,enc_gen_cat_map_grid,enc_gen_cat_map_blocks);
  
    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nRotated image = ");
      common::printArray8(enc_vec,total * channels);
    }
  
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      cv::Mat img_reshape(m,n,CV_8UC4,enc_vec);
      cv::imwrite(config::rotated_image_path,img_reshape);  
    }
 } 
  
  if(ROW_COL_SWAPPING == 1)
  {
    /*Row and Column Swapping*/
    cout<<"\nIn row and column swapping";
    dim3 enc_row_col_swap_grid(m,n,1);
    dim3 enc_row_col_swap_blocks(channels,1,1);
    run_encRowColSwap(enc_vec,final_vec,row_swap_lut_vec,col_swap_lut_vec,enc_row_col_swap_grid,enc_row_col_swap_blocks);
  
    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nRotated and Swapped image = ");
      common::printArray8(final_vec,total * channels);
    }
  
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      cv::Mat img_reshape(m,n,CV_8UC4,final_vec);
      cv::imwrite(config::swapped_image_path,img_reshape);  
    }
  
  }

  return 0;
}
