/*2D Logistic Map*/
#include "randomfunctions.hpp"
#include <cmath>

#define UNZERO 0.00000001


void twodSineLogisticModulationMap(double x,double y,double myu,double randnum,int number);




/*void twodSineLogisticModulationMap(double x,double y,double myu,double randnum,int number)
{

  int i=0;
  for(int i=0;i<number;++i)
  {
    ;
  }

}*/


int main()
{
  double myu=0.0,myu1=0.0,myu2=0.0,lambda1=0.0,lambda2=0.0,x=0.0872217,y=0.98121761,randnum=0.0;
  int number=2000;
  
  myu=0.8128190121;
  randnum=getRandomNumber(LOWER_LIMIT,UPPER_LIMIT);
  
  myu1 = getRandomNumber(2.75,3.40);
  myu2 = getRandomNumber(2.75,3.45);
  lambda1 = getRandomNumber(0.15,0.21);
  lambda2 = getRandomNumber(0.13,0.15);
  x = getRandomNumber(0,1);
  y = getRandomNumber(0,1);
  
  printf("\nx= %F",x);
  printf("\ny= %F",y);
  printf("\nmyu1= %F",myu1);
  printf("\nmyu2= %F",myu2);
  printf("\nlambda1= %F",lambda1);
  printf("\nlambda2= %F",lambda2);
  printf("\nrandnum= %F",randnum);
  cout<<"\nnumber= "<<number;
  
  //twodLogisticMapAdvanced(x,y,myu1,myu2,lambda1,lambda2,randnum,number);
  twodLogisticAdjustedSineMap(x,y,myu,number);  
  
  return 0;
}

