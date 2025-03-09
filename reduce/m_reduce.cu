#include<stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;


#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void print_matrix(float* matrix,int length,int width){
    for(int i = 0;i<width;i++){
        for(int j=0;j<length;j++){
            int addr = i*length + j;
            printf("%.1f, ",matrix[addr]);
        }
        printf("\n");
    }
}


__global__ void m_reduce(float* __restrict__ A,const int size,float* __restrict__ B){

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    const int tid = bx*blockDim.x+tx;

    const int s_size = 256;

    __shared__ float shareA[s_size];

    if(tx<size){
        shareA[tx] = A[tx];
    }else{
        shareA[tx] = 0;
    }

    __syncthreads();

    #pragma unroll
    for(int p = s_size/2;p>=32;p>>=1){
        if(tx<p){
            shareA[tx] += shareA[tx+p];
        }

        __syncthreads();
    }


    if(tx<32){
        float val = shareA[tx];
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        if(tx == 0){
            B[bx] = val;
            
        }
    }
}



void __global__ reduce_cp(float* __restrict__ A,float* __restrict__ B,const int size)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ float s_y[];

    float y = 0.0;
    const int stride = blockDim.x * gridDim.x;

    for (int n = bid * blockDim.x + tid; n < size; n += stride)
    {
        y += A[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    #pragma unroll
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        B[bid] = y;
    }
}


float reduce(float* A,int size){

    int blocksize = 256;
    int lastCsize = (size+blocksize-1)/blocksize;
    int Csize = 0;
    float* a = A;
    float* result;
    result = (float*)malloc(sizeof(float));

    bool start = false;
    int index = 1;
    float *C;
    while(1){
        // printf("reduce cycle:%d,lastCsize:%d,Csize:%d\n",index,lastCsize,Csize);
        if(!start){
            cudaMalloc(&C,lastCsize*sizeof(float));
        
            m_reduce<<<lastCsize,blocksize>>>(a,size,C);

            if(lastCsize == 1){
                break;
            }
            Csize = lastCsize;
            lastCsize = (lastCsize+blocksize-1)/blocksize;
            a = C ;
            start = true;
        }else{
            cudaMalloc(&C,lastCsize*sizeof(float));
        
            m_reduce<<<lastCsize,blocksize>>>(a,Csize,C);
            if(lastCsize == 1){
                cudaFree(a);
                break;
            }
            Csize = lastCsize;
            lastCsize = (lastCsize+blocksize-1)/blocksize;
            cudaFree(a);
            a = C ;
        }
        index++;

    }
    cudaMemcpy(result,C,sizeof(float),cudaMemcpyDeviceToHost);
    float res = *result;
    free(result);
    return res;
}



float book_reduce(float* A,int size){

    int blocksize = 256;
    int gridsize = 64;

    float* a = A;
    float* result;
    result = (float*)malloc(sizeof(float));

    float* B;
    float* C;
    cudaMalloc(&B,gridsize*sizeof(float));
    cudaMalloc(&C,sizeof(float));

    reduce_cp<<<gridsize,blocksize,blocksize>>>(A,B,size);
    reduce_cp<<<1,gridsize,gridsize>>>(B,C,gridsize);

    cudaMemcpy(result,C,sizeof(float),cudaMemcpyDeviceToHost);
    float res = *result;
    free(result);
    return res;
}



void test(){

    float* h_A;
    int M = 512;
    int N = 512;
    int size = M*N;


    h_A = (float*)malloc(size*sizeof(float));
    for(int i = 0;i<size;i++){
        h_A[i] = 1;
    }


    float* d_A;
    cudaMalloc(&d_A,size*sizeof(float));
    cudaMemcpy(d_A,h_A,size*sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    float res = reduce(d_A,size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);
    printf("m_reduce elasped_time:%.4f\n",elapsed_time);
    printf("m_reduce compute sum:%f,true compute sum:%d\n",res,size);




    cudaEventRecord(start);
    cudaEventQuery(start);

    res = book_reduce(d_A,size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time,start,stop);
    printf("book_reduce elasped_time:%.4f\n",elapsed_time);
    printf("book_reduce compute sum:%f,true compute sum:%d\n",res,size);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    cudaFree(d_A);

}


int main(){

    for(int i=0;i<5;i++){
        test();
    }



    return 0;
}