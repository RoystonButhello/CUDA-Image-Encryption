#ifndef RANDOMFUNCTIONS_H
#define RANDOMFUNCTIONS_H

#include <iostream>
#include <ctime>
#include <cstdio>
#include <random>
#include <chrono>
#include <cstdint>
#define LOWER_LIMIT 0.000001
#define UPPER_LIMIT 0.09
#define BIT_RETURN(A,LOC) (( (A >> LOC ) & 0x1) ? 1:0)
#define NUMBER_OF_BITS 16

using namespace std;

double getRandomNumber(double lower_limit,double upper_limit);
void twodLogisticMapBasic(double x,double y,double myu,double randnum,int number);
void twodLogisticMapAdvanced(double *&x, double *&y, uint16_t *&random_array, double myu1, double myu2, double lambda1, double lambda2,double randnum,int number);
void twodLogisticAdjustedSineMap(double *&x, double *&y, uint16_t *&random_array, double myu, uint32_t total);
void MTMap(uint16_t *&random_array,int number,int lower_limit,int upper_limit);
void grayLevelTransform(uint8_t *&img_vec,uint16_t *random_array,uint32_t total);
void show_ieee754 (double f);
void print_int_bits(int num);
uint16_t get_n_mantissa_bits_safe(double f,int number_of_bits);


double getRandomNumber(double lower_limit,double upper_limit)
{
  
   std::random_device r;
   std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
   mt19937 seeder(seed);
   //uniform_int_distribution<int> intGen(1, 32);
   uniform_real_distribution<double> realGen(lower_limit, upper_limit);   
   auto randnum=realGen(seeder);
   return (double)randnum;
}

void twodLogisticMapBasic(double x,double y,double myu,double randnum,int number)
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

void twodLogisticMapAdvanced(double *&x, double *&y, uint16_t *&random_array, double myu1, double myu2, double lambda1, double lambda2,double randnum,int number)
{
  printf("\n In 2DLMA");
  int i = 0;
  for(i = 0; i < number; ++i)
  {
    printf("\nx= %F",x[i]);
    x[i + 1] = x[i] * myu1 * (1 - x[i]) + lambda1 * (y[i] * y[i]);
    y[i + 1] = y[i] * myu2 * (1 - y[i]) + lambda2 * ((x[i] * x[i]) + x[i] * y[i]); 
  }
}

void twodLogisticAdjustedSineMap(double *&x, double *&y, uint16_t *&random_array, double myu, uint32_t total)
{
  printf("\nIn 2dLASM");
  int i=0;

  for(i = 0; i < total * 3; ++i)
  {
   
    //printf("\nx= %F",x[i]);
    random_array[i] = get_n_mantissa_bits_safe(x[i],NUMBER_OF_BITS);
    printf("\n%d",random_array[i]);
    x[i + 1] = sin(M_PI * myu * (y[i] + 3) * x[i] * (1 - x[i]));
    y[i + 1] = sin(M_PI * myu * (x[i + 1] + 3) * y[i] * (1 - y[i]));
    
    
  }
}

void MTMap(uint16_t *&random_array,int number,int lower_limit,int upper_limit)
{
    cout<<"\nIn MTMap";
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 seeder(seed);
    
    std::uniform_int_distribution<int> intGen(lower_limit,upper_limit);
 
    /* generate ten random numbers in [1,6] */
    for (size_t i = 0; i < number; ++i)
    {
        auto random_number=intGen(seeder);
        random_array[i]=(uint16_t)random_number;
    }
}

void grayLevelTransform(uint8_t *&img_vec,uint16_t *random_array,uint32_t total)
{
  for(int i = 0; i < total * 3; ++i)
  {
    img_vec[i] = img_vec[i] ^ random_array[i];
  }
  
}

/* formatted output of ieee-754 representation of float */
void show_ieee754 (double f)
{
    union {
        double f;
        uint32_t u;
    } fu = { .f = f };
    int i = sizeof f * CHAR_BIT;

    printf ("  ");
    while (i--)
        printf ("%d ", BIT_RETURN(fu.u,i));

    putchar ('\n');
    printf (" |- - - - - - - - - - - - - - - - - - - - - - "
            "- - - - - - - - - -|\n");
    printf (" |s|      exp      |                  mantissa"
            "                   |\n\n");
}

//Print bits of an integer
void print_int_bits(int num)
{   int x=1;
   for(int bit=(sizeof(int)*8)-1; bit>=0;bit--)
   {
      /*printf("%i ", num & 0x01);
      num = num >> 1;*/
      printf("%c",(num & (x << bit)) ? '1' : '0');
   }
}

uint16_t get_n_mantissa_bits_safe(double f,int number_of_bits)
{
    union {
        double f;
        uint32_t u;
    } fu = { .f = f };
    //int i = sizeof f * CHAR_BIT;
    int i=number_of_bits;
    int bit_store_32=0;
    uint8_t bit_store_8=0;
    uint16_t bit_store_16=0;
    
    //printf("\nBefore assigining any bits to bit_store_32=\n");
    //print_int_bits(bit_store_16);

    while (i--)
    {
        
        if(BIT_RETURN(fu.u,i)==1)
        {
            bit_store_16 |= 1 << i;
        }
        
    }
    
    
    //printf("\nAfter assigining bits to bit_store_32=\n");
    //print_int_bits(bit_store_16);
    

    //bit_store_8=(uint8_t)bit_store_32;
    return bit_store_16;
}


#endif
