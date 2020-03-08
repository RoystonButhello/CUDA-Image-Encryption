#include <iostream> /*For IO*/
#include <cstdlib>  /*For malloc()*/
#include <cuda.h>   /*For CUDA*/
#include <cuda_runtime.h>
#include <curand.h> /*For CURAND*/
#include <curand_kernel.h>
#include <cstdio>   /*For printf()*/
#include <cstdint>  /*For standard variable support*/

using namespace std;

int main()
{
  int n=10;
  int i;
  curandGenerator_t gen;
  unsigned int *devData, *hostData;
  
  /*Allocate n integers on the host*/
  hostData=(unsigned int*)malloc(n * sizeof(unsigned int));
  
  /*Initializing hostData*/
  for(int i=0;i<n;++i)
  {
    hostData[i]=0;
  }
  
  /*Allocate n unsigned ints on device*/
  cudaMalloc((void**) &devData , n * sizeof(unsigned int));
  
  /*Create an MTGP Host Generator*/
  curandCreateGenerator(&gen , CURAND_RNG_PSEUDO_MTGP32);
  
  /*Set seed*/
  curandSetPseudoRandomGeneratorSeed(gen , 1);

  /*Generate n ints on the device*/
  curandGenerate(gen, devData, n);
  
  /*Copy device memory to host*/
  cudaMemcpy(hostData, devData, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
  cout<<"\nRandom draws: \n";
  for(i = 0; i < n; ++i) 
  {
    printf(" %d",hostData[i]%1024);
  }
  cout<<"\n";
  
  /*Clean up*/
  curandDestroyGenerator(gen);
  
  return 0;
}

