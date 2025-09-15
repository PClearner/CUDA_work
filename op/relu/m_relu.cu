#include<stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
using namespace cooperative_groups;
const int M=1024;
const int N=1024;
float* answer = (float*)malloc(M*N*sizeof(float));

__global__ void cuda_relu(float* cudaptr,float* cudares,int M,int N){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    cudares[tid] = cudaptr[tid]>0? cudaptr[tid] : 0;

}

void cuda_relu_interface(float* output,float* cudaptr,float* cudares,int M,int N){
    int blocksize = N;
    int gridsize = M;

    cuda_relu<<<gridsize,blocksize>>>(cudaptr,cudares,M,N);    
    cudaMemcpy(output,cudares,M*N*sizeof(float),cudaMemcpyDeviceToHost);
}


void cpu_relu(float* input,float* output,int M,int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            output[i*N+j] = input[i*N+j]>0? input[i*N+j] : 0; 
        }
    }
}

void checkcompute(float* a, float* b, int size){
    const float epsilon = 1e-6f;  // 可以根据需要调整
    bool has_error = false;
    
    for(int i = 0; i < size; i++){
        float diff = fabsf(a[i] - b[i]);
        if(diff > epsilon) {
            printf("error!! ||answer[%d]:%.9f,result[%d]:%.9f\n", i, a[i], i, b[i]);
            printf("差值: %.9e (超过epsilon: %.9e)\n", diff, epsilon);
            has_error = true;
            return;
        }
    }
    
    if(!has_error) {
        printf("success (在epsilon=%.9f误差范围内)\n", epsilon);
    }
}

void print_matrix(float* matrix,int length,int width){
    for(int i = 0;i<width;i++){
        for(int j=0;j<length;j++){
            int addr = i*length + j;
            printf("%f, ",matrix[addr]);
        }
        printf("\n");
    }
}

void test(int j){

    float* input = (float*)malloc(M*N*sizeof(float));
    for(int i =0;i<M*N;i++){
        input[i] = i-(M/2)*N;
    }

    float* result = (float*)malloc(M*N*sizeof(float));
    // cudaMalloc(&cudaptr,M*N*sizeof(float));
    // cudaMalloc(&output,M*N*sizeof(float));
    // cudaMemcpy(cudaptr,flash,M*N*sizeof(float));
    float* cudaptr;
    float* cudares;
    cudaMalloc(&cudaptr,M*N*sizeof(float));
    cudaMalloc(&cudares,M*N*sizeof(float));
    cudaMemcpy(cudaptr,input,M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);
    
    if(j==0){
        cpu_relu(input,answer,M,N);
    }else if(j==1){
        cuda_relu_interface(result,cudaptr,cudares,M,N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);
    if(j==0){
        printf("cpu_relu elapsed_time:%.4f\n",elapsed_time);
    }else if(j==1){
        printf("cuda_relu elapsed_time:%.4f\n",elapsed_time);
        checkcompute(answer,result,M*N);
    }
    free(result);
    cudaFree(cudaptr);
    cudaFree(cudares); 




}

int main(){

    printf("M:%d,N:%d\n",M,N);
    for(int i=0;i<3;i++){
        printf("round:%d\n",i);
        for(int j=0;j<2;j++){
            test(j);
        }
    }
    // print_matrix(answer,N,M);
    return 0;
}


