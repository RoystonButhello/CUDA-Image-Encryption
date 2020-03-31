#ifndef PATTERN_H
#define PATTERN_H

namespace pattern 
{
  static inline void twodLogisticMapBasic(double x,double y,double myu,double randnum,int number);
  static inline void twodLogisticMapAdvanced(double *&x, double *&y, uint16_t *&random_array, double myu1, double myu2, double lambda1, double   lambda2,double randnum,int number);
  static inline void twodLogisticAdjustedSineMap(double *&x, double *&y, uint16_t *&random_array, double myu, uint32_t total);
  static inline void MTSequence(uint32_t *&random_array,uint32_t total,int lower_limit,int upper_limit,int seed);
  static inline void twodSineLogisticModulationMap(double *&x, double *&y, double alpha, double beta, uint32_t total);

  static inline void twodLogisticMapBasic(double x,double y,double myu,double randnum,int number)
  {
    int i=0;
  
    for(i=0;i<number;++i)
    {
      printf("\nx =%F",x);
      //printf("\ny =%F",y);
      if(x==0)
      {
        x = randnum;
        randnum = randnum + LOWER_LIMIT; 
      }  
    
        x = myu * ((3 * y) + 1) * x * (1 - x);
        y = myu * (3 * x + 1) * y * (1 - y);
     }

  }

  static inline void twodLogisticMapAdvanced(double *&x, double *&y, uint16_t *&random_array, double myu1, double myu2, double lambda1, double lambda2,double randnum,int number)
  {
    printf("\n In 2DLMA");
    int i = 0;
    for(i = 0; i < number; ++i)
    {
      //printf("\nx= %F",x[i]);
      x[i + 1] = x[i] * myu1 * (1 - x[i]) + lambda1 * (y[i] * y[i]);
      y[i + 1] = y[i] * myu2 * (1 - y[i]) + lambda2 * ((x[i] * x[i]) + x[i] * y[i]); 
    }
  }

  static inline void twodLogisticAdjustedSineMap(double *&x, double *&y, uint16_t *&random_array, double myu, uint32_t total)
  {
    printf("\nIn 2dLASM");
    int i=0;

    for(i = 0; i < (total * 3) - 1; ++i)
    {
   
      //printf("\nx= %F",x[i]);
      random_array[i] = common::get_n_mantissa_bits_safe(x[i],NUMBER_OF_BITS);
      //printf("\n%d",random_array[i]);
      x[i + 1] = sin(M_PI * myu * (y[i] + 3) * x[i] * (1 - x[i]));
      y[i + 1] = sin(M_PI * myu * (x[i + 1] + 3) * y[i] * (1 - y[i]));
    
    
    }
  }

  static inline void MTSequence(uint32_t *&random_array,uint32_t total,int lower_limit,int upper_limit,int seed)
  {
    cout<<"\nIn MTMap";
    //std::random_device r;
    //std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 seeder(seed);
    
    std::uniform_int_distribution<int> intGen(lower_limit,upper_limit);
 
    /* generate ten random numbers in [1,6] */
    for (size_t i = 0; i < total * 3; ++i)
    {
        auto random_number=intGen(seeder);
        random_array[i]=(uint32_t)random_number;
    }
  }

  static inline void twodSineLogisticModulationMap(double *&x, double *&y, double alpha, double beta, uint32_t total)
  {
    for(int i = 0; i < (total * 3) - 1; ++i)
    {
      x[i + 1] = alpha * (sin(M_PI * y[i]) + beta) * x[i] * (1 - x[i]);
      y[i + 1] = alpha * (sin(M_PI * x[i + 1]) + beta) * y[i] * (1 - y[i]);
    }
  }

}


#endif
