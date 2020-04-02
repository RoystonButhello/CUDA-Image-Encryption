#ifndef SERIAL_H
#define SERIAL_H

namespace serial
{
  static inline void xorImageEnc(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position);
  static inline void xorImageDec(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position);
  static inline void grayLevelTransform(uint8_t *&img_vec,uint32_t *random_array,uint32_t total);
  
    
  static inline void generateRelocationVec(uint32_t *&rotation_vector,int a,int b,int c,int offset,double x,double y,uint32_t m,uint32_t n,uint32_t total);
  
  
  static inline void columnRotator(cv::Mat3b &img, cv::Mat3b col, int index, int offset, int m);
  static inline void rowRotator(cv::Mat3b &img, cv::Mat3b row, int index, int offset, int n);

  static inline void xorImageEnc(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position)
  { 
     //cout<<"\nIn xorImageEnc";
     uint32_t total=m*n;
     /*for(int i=0;i<total*3;++i)
     {
       img_xor_vec[i]=0;
     }*/

     int cnt=0;
     //printf("\n %d = %d ^ %d",img_xor_vec[0],img_vec[0],INIT);
     for(int i=xor_position;i>0;--i)
     {
       ++cnt;
       img_xor_vec[i]=img_vec[i] ^ img_xor_vec[i-1];
       //printf("\n %d = %d ^ %d",img_xor_vec[i],img_vec[i],img_xor_vec[i-1]);
     } 
  
     img_xor_vec[0]=img_xor_vec[0] ^ img_xor_vec[(total*3)-1];
   
     for(int i=1;i<(total*3)-cnt;++i)
     {
       img_xor_vec[i]=img_vec[i] ^ img_xor_vec[i-1];
     }

  }

  static inline void xorImageDec(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position)
  { 
     //cout<<"\nIn xorImageDec";
     uint32_t total=m*n;
     /*for(int i=0;i<total*3;++i)
     {
       img_xor_vec[i]=0;
     }*/

     int cnt=0;
   
  
     //printf("\n %d = %d ^ %d",img_xor_vec[0],img_vec[0],INIT);
     for(int i=xor_position;i>0;--i)
     {
       ++cnt;
       img_xor_vec[i]=img_vec[i] ^ img_vec[i-1];
       //printf("\n %d = %d ^ %d",img_xor_vec[i],img_vec[i],img_vec[i-1]);
     } 

     img_xor_vec[0]=img_vec[0] ^ img_xor_vec[(total*3)-1];
   
     for(int i=1;i<=(total*3)-cnt;++i)
     {
       img_xor_vec[i]=img_vec[i] ^ img_vec[i-1];
       //printf("\n %d = %d ^ %d",img_xor_vec[i],img_vec[i],img_vec[i-1]);
     }  
  }


  static inline void grayLevelTransform(uint8_t *&img_vec,uint32_t *random_array,uint32_t total)
  {
    for(int i = 0; i < total * 3; ++i)
    {
      img_vec[i] = img_vec[i] ^ random_array[i];
    }
  
  }

  static inline void rowColSwapEnc(cv::Mat3b &img_in,cv::Mat3b &img_out,uint32_t *&rowSwapLUT,uint32_t *&colSwapLUT,uint32_t m,uint32_t n,uint32_t total)
  {
    int get_row = 0, get_col = 0;
    int row_constant = (m * 3);
    int row = 0,col = 0;
    int element_index = 0;
  
    int i = 0, j = 0, k = 0;
    for(i = 0; i < m; ++i)
    {
      row = rowSwapLUT[i];
    
      for(j = 0; j < n; ++j)
      {
        col = colSwapLUT[j];
        int pixel_index_in = i * n + j;
        int pixel_index_out = row * n + col;
        //printf("\n%d",i * m + j);
        img_out.at<Vec3b>(pixel_index_in) = img_in.at<Vec3b>(pixel_index_out);
        
       }
     }
  }

  static inline void rowColSwapDec(cv::Mat3b &img_in,cv::Mat3b &img_out,uint32_t *&rowSwapLUT,uint32_t *&colSwapLUT,uint32_t m,uint32_t n,uint32_t total)
  {
    int get_row = 0, get_col = 0;
    int row_constant = (m * 3);
    int row = 0,col = 0;
    int element_index = 0;
    //printf("\nm = %d",m);
    //printf("\nn = %d",n);
    for(int i = 0; i < m; ++i)
    {
      row = rowSwapLUT[i];
      for(int j = 0; j < n; ++j)
      {
        col = colSwapLUT[j];
        int pixel_index_in = i * n + j;
        //printf("\npixel_index_in = %d",pixel_index_in);
        int pixel_index_out = row * n + col;
        img_out.at<Vec3b>(pixel_index_out) = img_in.at<Vec3b>(pixel_index_in);
        
      }
    }
   }
  
  static inline void generateRelocationVec(uint32_t *&rotation_vector,int a,int b,int c,int offset,double x,double y,uint32_t m,uint32_t n,uint32_t total)
  {
    uint64_t exponent = (uint64_t)pow(10,14);
    double unzero = 0.0000000001;
    double *random_vector = (double*)malloc(sizeof(double) * n);
    
    /*Skipping 1st offset values*/
    for(int i = 0; i < offset; ++i)
    {
      x = fmod((x + a * y),1) + unzero;
      y = fmod((b * x + c * y),1) + unzero; 
    }
  
    /*Generating random vector*/
    for(int i = 0; i < (total / 2); ++i)
    {
      x = fmod((x + a * y),1) + unzero;
      y = fmod((b * x + c * y),1) + unzero;
      random_vector[2 * i] = x;
      random_vector[(2 * i) + 1] = y;
    }
    
    /*Generating rotation vector*/
    for(int i = 0; i < n; ++i)
    {
      rotation_vector[i] = (uint32_t)fmod((random_vector[i] * exponent),m);
    }
  
  }
  
  static inline void columnRotator(cv::Mat3b &img, cv::Mat3b col, int index, int offset, int m)
  {
    
    if (offset > 0)
    {
        for (int k = 0; k < m; k++)
        {
            img.at<Vec3b>(k, index) = col.at<Vec3b>((k + offset) % m, 0);
        }
    }
    
    else if (offset < 0)
    {
        for (int k = 0; k < m; k++)
        {
            img.at<Vec3b>(k, index) = col.at<Vec3b>((k + offset + m) % m, 0);
        }
    }
    
    else
    {
        for (int k = 0; k < m; k++)
        {
            img.at<Vec3b>(k, index) = col.at<Vec3b>(k, 0);
        }
    }
}

static inline void rowRotator(cv::Mat3b &img, cv::Mat3b row, int index, int offset, int n)
{
    // N elements per row
    if (offset > 0)
    {
        for (int k = 0; k < n; k++)
        {
            img.at<Vec3b>(index, k) = row.at<Vec3b>(0, (k + offset) % n);
        }
    }
    else if (offset < 0)
    {
        for (int k = 0; k < n; k++)
        {
            img.at<Vec3b>(index, k) = row.at<Vec3b>(0, (k + offset + n) % n);
        }
    }
    else
    {
        for (int k = 0; k < n; k++)
        {
            img.at<Vec3b>(index, k) = row.at<Vec3b>(0, k);
        }
    }
}  
 
}


#endif
