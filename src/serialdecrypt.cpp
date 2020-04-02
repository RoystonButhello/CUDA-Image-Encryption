#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"

int main()
{
  cv::Mat3b image;
  image = cv::imread("airplane_encrypted.png",cv::IMREAD_COLOR);
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
  
  uint8_t *enc_vec  = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint8_t *dec_vec  = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *row_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * m);
  uint32_t *col_swap_lut_vec = (uint32_t*)malloc(sizeof(uint32_t) * n);
  
      
  cv::Mat3b imgout(m,n);
  pattern::MTSequence(row_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  pattern::MTSequence(col_random_vec,total,config::lower_limit,config::upper_limit,config::seed_lut_gen);
  common::genLUTVec(row_swap_lut_vec,m);
  common::genLUTVec(col_swap_lut_vec,n);
  common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);    
  
  /*if(DEBUG_VECTORS == 1)
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
  }*/  


    //Unpermutation
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
  
   
   image = image.reshape(1,image.rows * image.cols);
   imgout = imgout.reshape(1,imgout.rows * imgout.cols);
   
  //Row and Column Swapping
  //serial::rowColSwapDec(image,imgout,row_swap_lut_vec,col_swap_lut_vec,m,n,total);
  image = image.reshape(3,m);
  imgout = imgout.reshape(3,m);
  
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nImage after decryption = ";
    common::printImageContents(image);
    cout<<"\n\nimgout in serialdecrypt";
    common::printImageContents(imgout);
  }
  
  cv::imwrite("airplane_decrypted.png",image);
  
  
  return 0;
}

