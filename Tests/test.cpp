#include <iostream>
#include <cstdio>
using namespace std;

void takeArray(int Arr[],int length)
{
  for(int i = 0; i < length; ++i)
  {
    Arr[i]= Arr[i] * 2;  
  }
  cout<<"\nsizeof(Arr) = "<<sizeof(Arr);
}
int main()
{
  int arr[4];
  for(int i = 0; i < 4; ++i)
  {
    arr[i] = i;
  }
  
  cout<<"\narr before call = ";
  for(int i = 0; i < 4; ++i)
  {
    printf(" %d",arr[i]);
  }
  
  takeArray(arr,4);
  
  cout<<"\narr after call = ";
  for(int i = 0; i < 4; ++i)
  {
    printf(" %d",arr[i]);
  }
  
  return 0;
}
