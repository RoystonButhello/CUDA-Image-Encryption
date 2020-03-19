#include <iostream>
#include <cstdio>
#define N 8
using namespace std;

void swapRows(int n,int swapRow[N],int x[N][N][3])
{
  cout<<"\nIn swapRows";
  int row=0;
  for(int i = 0; i < (n); i+=3)
  {
    row = i + 3;
    if(row > N - 3)
    {
      row = i;
    } 
    printf("\n %d , %d",i,row);
    //int temp = swapRow[i];
    //printf("\n%d , swapRow[%d] = %d",i,i,swapRow[i]);
    
    for(int j = 0; j < (n); j++)
    {
      //int temp2 = swapCol[j];
      for(int k = 0; k < 3; ++k)
      {
        
        std::swap(x[i][j][k],x[row][j][k]);
        
      }
    }
  }
}

void swapCols(int n,int swapCol[N],int x[N][N][3])
{
  cout<<"\nIn swapCols";
  for(int i = 0; i < (n); i++)
  {
    //int temp = swapRow[i];
    for(int j = 0; j < (n); j+=2)
    {
      //printf("\n %d , %d",j,j+1);
      //int temp2 = swapCol[j];
      for(int k = 0; k < 3; ++k)
      {
        
        std::swap(x[i][j][k],x[i][j+1][k]);
        
      }
    }
  }
}


int main()
{
  
  int x[N][N][3];
  
  int A[N * N * 3];
  int original[N]= {0,1,2,3};
  int swapRow[N] = {1,3,0,2};
  int swapCol[N] = {1,3,0,2};
  
  int cnt = 0;
  
  cout<<"\nOriginal Matrix";
  for(int i = 0; i < N; ++i)
  {
    cout<<"\n";
    for(int j = 0; j < N; ++j)
    {
      for(int k = 0; k < 3; ++k)
      {
        //printf("\nX[%d][%d][%d] = %d",i,j,k,x[i][j][k]);
        
        x[i][j][k]=cnt+1;
        //A[cnt] = x[i][j][k];
        ++cnt;
        printf(" %d",x[i][j][k]);
      }
    }
  }  
  
 swapRows(N,swapRow,x); 
 //swapCols(N,swapCol,x); 
  
  cout<<"\nAfter row swapping once ";
  for(int i = 0; i < N; ++i)
  {
    
    cout<<"\n\n";
    for(int j = 0 ; j < N; ++j)
    {
      for(int k = 0; k < 3; ++k)
      {
        printf(" %d",x[i][j][k]);
      }
    }
  }
  
 swapRows(N,swapRow,x);
 //swapCols(N,swapCol,x); 
  
  cout<<"\nAfter row swapping again ";
  for(int i = 0; i < N; i++)
  {
    
    cout<<"\n\n";
    for(int j = 0 ; j < N; ++j)
    {
      for(int k = 0; k < 3; ++k)
      {
        printf(" %d",x[i][j][k]);
      }
    }
  }
  return 0;
}
