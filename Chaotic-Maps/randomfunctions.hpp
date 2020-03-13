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

using namespace std;

double getRandomNumber(double lower_limit,double upper_limit);
void twodLogisticMapBasic(double x,double y,double myu,double randnum,int number);
void twodLogisticMapAdvanced(double x, double y, double myu1, double myu2, double lambda1, double lambda2,double randnum,int number);
void twodLogisticAdjustedSineMap(double x, double y, double myu, int number);

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

void twodLogisticMapAdvanced(double x, double y, double myu1, double myu2, double lambda1, double lambda2,double randnum,int number)
{
  printf("\n In 2DLMA");
  int i = 0;
  for(i = 0; i < number; ++i)
  {
    printf("\nx= %F",x);
    x = x * myu1 * (1 - x) + lambda1 * (y * y);
    y = y * myu2 * (1 - y) + lambda2 * ((x * x) + x * y); 
  }
}

void twodLogisticAdjustedSineMap(double x, double y, double myu, int number)
{
  printf("\nIn 2dLASM");
  int i=0;

  for(i=0;i<number;++i)
  {
    printf("\nx = %F",x);
    x = sin(M_PI * myu * (y + 3) * x * (1 - x));
    y = sin(M_PI * myu * (x + 3) * y * (1 - y));
    
  }
}


#endif
