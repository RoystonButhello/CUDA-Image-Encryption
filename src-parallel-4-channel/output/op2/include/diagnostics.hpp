#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

namespace diagnostics
{
  /*Function Prototypes*/
  static inline void checkImageVectors(int *plain_img_vec,int *decrypted_img_vec,uint32_t total);
  static inline void checkImageMatrices(cv::Mat3b plain_image,cv::Mat3b decrypted_image);


  static inline void checkImageVectors(int *plain_img_vec,int *decrypted_img_vec,uint32_t total)
  {
    int cnt=0,gt255=0;
    for(int i=0;i<total*3;++i)
    {
      if(decrypted_img_vec[i]-plain_img_vec[i]!=0)
      {
        ++cnt;
      }
    
      if(decrypted_img_vec[i]>255||plain_img_vec[i]>255)
      {
        ++gt255;
      }
    
  }
    printf("\nNumber of vector differences= %d",cnt);
    printf("\nNumber of gt255= %d",gt255);
  }

  static inline void checkImageMatrices(cv::Mat3b plain_image,cv::Mat3b decrypted_image)
  {
    int cnt=0;
     for(int i=0; i < plain_image.cols * plain_image.rows * 3; ++i)
      {
        
        if(plain_image.at<uint8_t>(i) - decrypted_image.at<uint8_t>(i) != 0)
        {
           ++cnt;
        }        
        
      
    }
      printf("\nNumber of matrix differences= %d",cnt);
  }
}
#endif
