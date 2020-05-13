#include <iostream> /*For IO*/
#include <cuda.h>   /*For CUDA*/
#include <cuda_runtime.h>
#include <curand.h> /*For curand*/
#include <curand_kernel.h>
#include <cstdio>

using namespace std;

/*Kernel to initialize n random number generators*/
__global__ void setup_kernel(curandState *state)
{
  int id = threadIdx.x + blockIdx.x * 64;
  
  /*Each thread gets the same seed, a different sequence number, no offset*/
  curand_init(1234, id, 0 &state[id]);
}

/*Kernel to generate random numbers*/
__global__ void generate_kernel(curandState *state, int *result)
{
  int id = threadIdx.x + blockIdx.x * 64;
  int cnt=0;
  unsigned int x;
  
  /*Copy RNG state to local memory for efficiency*/  
  curandState localState = state[id];
  
  /*Generate Random unsigned ints. Here we are sending the local state of each RNG to curand. If we don't, then CURAND will generate the same   number*/
  for(int n = 0; n < 100000; n++ )
  {
    x = curand(&localState);
    /*check if RN is odd*/
    if(x & i)
    {
      cnt++;
    }
  
   }
   
   /*Copy state back to global memory. We do this in order to ensure that the local state is updated for every kernel call. This allows each kernel call to have its own sequence of random numbers*/

results[id] += cnt;

}

int main()
{
  int i,total;
  int *devResults, *hostResults;
  curandState *devStates;
  
  /*Host mem for test results*/
  hostResults = (int*)malloc(64*64*sizeof(int));
  
  /*Allocate space for results on device*/
  cudaMalloc((void**)&devResults, 64*64*sizeof(int));
  
  /*Set results to 0*/
  cudaMemset(devResults, 0, 64 * 64 * sizeof(int));
  
  /*Allocate space for PRNG states on device*/
  cudaMalloc((void**)&devResults, 64*64*sizeof(curandState));
  
  /*Setup PRNG States*/
  setup_kernel<<<64,64>>>(devStates);

  /*Generate and use PRNs*/
  for(i = 0; i < 10; ++i)
  {
    generate_kernel<<<64,64>>>(devStates, devResults);
  }  
  
  /*Copy device memory to host*/
  cudaMemcpy(hostResults, devResults, 64 * 64 * sizeof(int),cudaMemcpyDeviceToHost);
  
  /*Show result*/
  total=0;
  
  for(i = 0; i < 64 * 64; ++i)
  {
    total += hostResults[i];  
  }
  
  printf("\nFraction odd was 10.13f\n", (float)total / 64.0f * 64.0f *1000.0f *10.0f);
  

  return 0;
}
