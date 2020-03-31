#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"

int main()
{
  cv::Mat image;
  image = cv::imread(config::input_image_path,cv::IMREAD_COLOR);
  if(!image.data)
  {
    cout<<"\nCould not load image from "<<config::input<<"\n Exiting...";
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
  uint8_t *img_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint8_t *enc_vec  = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint32_t *random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *row_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *col_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * n);
  uint32_t *U = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *V = (uint32_t*)malloc(sizeof(uint32_t) * n);
  
  /*Image Permutation Phase*/
  
  /*Row and Column Swapping*/
  common::flattenImage(image,img_vec);
  //serial::rowColSwapDec()  
  
  return 0;
}

