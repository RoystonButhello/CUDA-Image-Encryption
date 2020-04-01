#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"

int main()
{
  cv::Mat image;
  image = cv::imread(config::encrypted_image_path,cv::IMREAD_COLOR);
  if(!image.data)
  {
    cout<<"\nCould not load encrypted image from "<<config::encrypted_image_path<<"\n Exiting...";
    exit(0);
  }
  
  if(RESIZE_TO_DEBUG == 1)
  {
    cv::resize(image,image,cv::Size(config::rows,config::cols),CV_INTER_LANCZOS4);
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
  
  uint8_t *enc_vec  = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint8_t *dec_vec  = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *row_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *col_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * n);
  uint32_t *U = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *V = (uint32_t*)malloc(sizeof(uint32_t) * n);
  
  common::flattenImage(image,enc_vec);
  pattern::MTSequence(row_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  pattern::MTSequence(col_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  common::genLUTVec(row_swap_lut_vec,m);
  common::genLUTVec(col_swap_lut_vec,n);
      

  if(DEBUG_VECTORS == 1)
  {
    cout<<"\n\nOriginal image vector = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",enc_vec[i]);
    }
    
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
   
  /*Image Permutation Phase*/
  
  /*Row and Column Swapping*/
  
  common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);
  
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
  
  serial::rowColSwapDec(enc_vec,dec_vec,row_swap_lut_vec,col_swap_lut_vec,m,n,total);
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\nEncrypted image vector = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",enc_vec[i]);
    }
    
    cout<<"\nDecrypted image vector = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",dec_vec[i]);
    }

  }
  
  if(DEBUG_IMAGES == 1)
  {
   cv::Mat img_resize(m,n,CV_8UC3,dec_vec);
   cv::imwrite(config::decrypted_image_path,img_resize);
  }
    
  return 0;
}

