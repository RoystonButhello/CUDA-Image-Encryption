#ifndef SERIAL_H
#define SERIAL_H

/**
 * This header file contains the Gray Level Transform function used for image diffusion 
 */
namespace serial
{
  
  static inline void grayLevelTransform(uint8_t *&img_vec,uint32_t *random_array,uint32_t total);
  
  /**
   * Diffuses image vector with pseudorandom sequence
   */
  static inline void grayLevelTransform(uint8_t *&img_vec,uint32_t *random_array,uint32_t total)
  {
    int i = 0;
    for(i = 0; i < total; ++i)
    {
      img_vec[i] = img_vec[i] ^ random_array[i];
    }
    
  }

  
}


#endif
