#ifndef SERIAL_H
#define SERIAL_H

namespace serial
{
  static inline void xorImageEnc(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position);
  static inline void xorImageDec(uint8_t *&img_vec,uint8_t *&img_xor_vec,uint32_t m,uint32_t n,uint16_t xor_position);
  static inline void grayLevelTransform(uint8_t *&img_vec,uint32_t *random_array,uint32_t total);
  static inline void rowColSwapEnc(uint8_t *&img_in,uint8_t *&img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,uint32_t m,uint32_t n,uint8_t channels);
  static inline void rowColSwapDec(uint8_t *&img_in,uint8_t *&img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,uint32_t m,uint32_t n,uint8_t channels);


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
    int i = 0;
    for(i = 0; i < total; ++i)
    {
      img_vec[i] = img_vec[i] ^ random_array[i];
    }
    //printf("\n i after diffusion = %d",i);
  }

  static inline void rowColSwapEnc(uint8_t *&img_in,uint8_t *&img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,uint32_t m,uint32_t n,uint8_t channels)
  {
    cout<<"\nIn rowColSwapEnc";
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
        //printf("\n%d = %d",pixel_index_in,pixel_index_out);  
        //printf("\n%d",i * m + j);
        for(k = 0; k < channels; ++k)
        {
          int gray_level_index_in = pixel_index_in * channels + k;
          int gray_level_index_out = pixel_index_out * channels + k;
          img_out[gray_level_index_in] = img_in[gray_level_index_out];
          
          //printf("\n%d",pixel_index_in * 3 + k);
        } 
       }
     }
  }

  static inline void rowColSwapDec(uint8_t *&img_in,uint8_t *&img_out,uint32_t *rowSwapLUT,uint32_t *colSwapLUT,uint32_t m,uint32_t n,uint8_t channels)
  {
    cout<<"\nIn rowColSwapDec";
    int row = 0,col = 0;
    int element_index = 0;
    printf("\nm = %d",m);
    printf("\nn = %d",n);
    for(int i = 0; i < m; ++i)
    {
      row = rowSwapLUT[i];
      for(int j = 0; j < n; ++j)
      {
        col = colSwapLUT[j];
        int pixel_index_in = i * n + j;
        //printf("\npixel_index_in = %d",pixel_index_in);
        int pixel_index_out = row * n + col;
        //printf("\n%d = %d",pixel_index_out,pixel_index_in);
        for(int k = 0; k < channels; ++k)
        {
          int gray_level_index_in = pixel_index_in * channels + k;
          int gray_level_index_out = pixel_index_out * channels + k;
          
          //printf("\ngray_level_index_in = %d",gray_level_index_in);
          //printf("\ngray_level_index_out = %d",gray_level_index_out);
          img_out[gray_level_index_out] = img_in[gray_level_index_in];
        
          //if(img_out[gray_level_index_out] == 0)
          //{
              //printf("\nglio %d = %d * 3 + %d",gray_level_index_out,pixel_index_out,k);
             //printf("\n\nglii %d = %d * 3 + %d",gray_level_index_in,pixel_index_in,k);
            //printf("\n\n\nimg_in = %d",img_in[gray_level_index_in]);
          //} 
        
        }
      }
    }
   }

}


#endif
