#include<stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
using namespace cooperative_groups;

const int M=1024;
const int N=1024;

float* answer = (float*)malloc(M*N*sizeof(float));

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        float old_float = __int_as_float(assumed);
        float new_float = fmaxf(old_float, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_float));
    } while(assumed != old);
    
    return __int_as_float(old);
}

__global__ void softmax(float* input,float* output,int M,int N){
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    extern __shared__ float mem[];
    mem[tid] = input[tid+bx*N];
    float* maxstore = &mem[N];
    
    float* sumstore = &mem[N+1];
    if(tid==0){
        *maxstore=-INFINITY;
        *sumstore = 0;
    }
    float max=mem[tid];
    __syncthreads();

    atomicMaxFloat(maxstore,max);
    __syncthreads();

    atomicAdd(sumstore,expf(mem[tid]-*maxstore));
    __syncthreads();

    output[bx*N + tid] = expf(mem[tid]-(*maxstore))/(*sumstore);
}

void softmax_interface(float* input,float* output,float* cudaptr,float* cudares,int M,int N){
    int blocksize = N;
    int gridsize = M;

    softmax<<<gridsize,blocksize,(N+2)*sizeof(float)>>>(cudaptr,cudares,M,N);
    cudaMemcpy(output,cudares,M*N*sizeof(float),cudaMemcpyDeviceToHost);
}

__global__ void softmax_flash(float* input,float* output,int M,int N){
    int tid = blockIdx.x;
    extern __shared__ float mem[];
    float* maxstore = &mem[N];
    float* sumstore = &mem[N+1];
    *maxstore = -INFINITY;
    *sumstore = 0;
    int f4num = N/4;
    int f4remain = N%4;  
    // float lastmax = -INFINITY;
    float max = -INFINITY;
    for(int i=0;i<f4num;i++){
        float4* tmp = (float4*)(input+i*4 + tid*N);
        *(float4*)(mem+i*4) = *tmp;
        if(tmp->x>max){
            max = tmp->x;
        }
        if(tmp->y>max){
            max = tmp->y;
        }
        if(tmp->z>max){
            max = tmp->z;
        }
        if(tmp->w>max){
            max = tmp->w;
        }
        *sumstore = *sumstore*expf((*maxstore)- max) + expf(tmp->x - max) + 
                    expf(tmp->y - max) +expf(tmp->z-max)+expf(tmp->w-max);
        *maxstore = max;
    }

    for(int i=N-f4remain;i<N;i++){
        mem[i] = input[i+tid*N];
        max = max > mem[i] ? max : mem[i];
        *sumstore = *sumstore*expf((*maxstore)- max) + expf(mem[i] - max);
        *maxstore = max;
    }

    for(int i=0;i<f4num;i++){
        float4* tmp1 = (float4*)(mem+i*4);
        // (float4*)(mem+i*4) = tmp;
        float4* tmp2 = (float4*)(output+i*4+tid*N);

        tmp2->x = expf(tmp1->x - (*maxstore))/(*sumstore);
        tmp2->y = expf(tmp1->y - (*maxstore))/(*sumstore);
        tmp2->z = expf(tmp1->z - (*maxstore))/(*sumstore);
        tmp2->w = expf(tmp1->w - (*maxstore))/(*sumstore);
    }

    for(int i=N-f4remain;i<N;i++){
        output[i + tid*N] = expf(mem[i]-(*maxstore))/(*sumstore);
    }
}



void softmax_flash_interface(float* input,float* output,float* cudaptr,float* cudares,int M,int N){
    int blocksize = 1;
    int gridsize = M;

    softmax_flash<<<gridsize,blocksize,(N+2)*sizeof(float)>>>(cudaptr,cudares,M,N);
    cudaMemcpy(output,cudares,M*N*sizeof(float),cudaMemcpyDeviceToHost);
}

void cpu_softmax_flash(float* input,float* output,int M,int N){
    for(int i=0;i<M;i++){
        float lastmax = -INFINITY;
        float lastsum = 0.f;
        for(int j=0;j<N;j++){
            float max = input[i*N + j] > lastmax? input[i*N + j] : lastmax;
            lastsum = lastsum*expf(lastmax-max) + expf(input[i*N + j]-max);
            lastmax = max;
        }
        for(int j=0;j<N;j++){
            output[i*N + j] = expf(input[i*N + j]-lastmax) / lastsum;
        }
    }
}

void cpu_softmax(float* input,float* output,int M,int N){
    for(int i=0;i<M;i++){
        float sum = 0.f;
        float max = -INFINITY;
        for(int j=0;j<N;j++){
            max = input[i*N + j] > max? input[i*N + j] : max;
        }

        for(int j=0;j<N;j++){
            sum+=expf(input[i*N + j]-max);
        }

        for(int j=0;j<N;j++){
            output[i*N + j] = expf(input[i*N + j]-max)/sum;
        }
    }
}

// void checkcompute(float* a,float* b,int size){
//     for(int i=0;i<size;i++){
//         if(a[i]!=b[i]){
//             printf("error!!   ||answer[%d]:%9f,result[%d]:%9f\n",i,a[i],i,b[i]);
//             return;
//         }
//     }
//     printf("success\n");
// }

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

void test(int j){

    float* input = (float*)malloc(M*N*sizeof(float));
    for(int i =0;i<M*N;i++){
        input[i] = i;
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
        cpu_softmax(input,answer,M,N);
    }else if(j==1){
        cpu_softmax_flash(input,result,M,N);
    }else if(j==2){
        softmax_interface(input,result,cudaptr,cudares,M,N);
    }else if(j==3){
        softmax_flash_interface(input,result,cudaptr,cudares,M,N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);
    if(j==0){
        printf("cpu_softmax elapsed_time:%.4f\n",elapsed_time);
    }else if(j==1){
        printf("cpu_softmax_flash elapsed_time:%.4f\n",elapsed_time);
        checkcompute(answer,result,M*N);
    }else if(j==2){
        printf("cuda_softmax elapsed_time:%.4f\n",elapsed_time);
        checkcompute(answer,result,M*N);
    }else if(j==3){
        printf("cuda_softmax_flash elapsed_time:%.4f\n",elapsed_time);
        checkcompute(answer,result,M*N);        
    }
    free(result);
    cudaFree(cudaptr);
    cudaFree(cudares); 
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

int main(){
    printf("M:%d,N:%d\n",M,N);
    for(int i=0;i<3;i++){
        printf("round:%d\n",i);
        for(int j=0;j<4;j++){
            test(j);
        }
    }
    // print_matrix(answer,N,M);
    // printf("answer[0]:%f\n",answer[0]);
}

