#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

__global__ void generate( curandState* globalState, int * result, int *max, int *min, int count ) 
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState; 

    if (ind < count)

        result[ind] = truncf(*min +(*max - *min)*RANDOM);
}

int main( int argc, char** argv) 
{
    int N = 32; // no of random numbers to be generated

    int MIN = 10; // max range of random number
    int MAX = 100; // min range of random number

    dim3 tpb(N,1,1);
    curandState* devStates;
    cudaMalloc ( &devStates, N*sizeof( curandState ) );

    // setup seeds
    setup_kernel <<< 1, tpb >>> ( devStates, time(NULL) );

    int *d_result, *h_result;

    cudaMalloc(&d_result, N * sizeof(int));
    h_result = (int *)malloc(N * sizeof(int));

    int *d_max, *h_max, *d_min, *h_min;

    cudaMalloc(&d_max, sizeof(int));
    h_max = (int *)malloc(sizeof(int));

    cudaMalloc(&d_min, sizeof(int));
    h_min = (int *)malloc(sizeof(int));

    *h_max =MAX;
    *h_min =MIN;

    cudaMemcpy(d_max, h_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, h_min, sizeof(int), cudaMemcpyHostToDevice);

    // generate random numbers
    generate <<< 1, tpb >>> ( devStates, d_result, d_max, d_min, N );

    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

      for (int i = 0; i < N; i++)
    printf("random number= %d\n", h_result[i]);

    return 0;
}
