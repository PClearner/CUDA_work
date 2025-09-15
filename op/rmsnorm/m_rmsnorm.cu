#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
using namespace cooperative_groups;
const int M=1024;
const int N=1024;
float* answer = (float*)malloc(M*N*sizeof(float));

__global__ void cuda_rms(float* cudaptr,float* cudares,float* cuda_weight,int M,int N,float t){
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int f4num = (N+3)/4;
    int f4remain = N%4;
    // int threadpernum = ((N+3)/4)/blockDim.x;9


    // int index = threadpernum*tid + bx*N;
    float sum = 0.f;
    extern __shared__ float mem[];

    if(N<32){
        if(tid<N){
            mem[tid] = cudaptr[bx*N+tid];
            mem[N+tid] = cuda_weight[tid];
        }
    }else{
        for(int i = tid;i<f4num;i+=32){
            if(i*4+4<=N){
                // float4 tmp = *(float4*)(cudaptr+bx*N+i*4);
                *(float4*) (mem+i*4) = *(float4*)(cudaptr+bx*N+i*4);
                // tmp = (float4*)(cuda_weight+i*4);
                *(float4*) (mem+N+i*4) = *(float4*)(cuda_weight+i*4);
            }else{
                for(int j=0;j<f4remain;j++){
                    mem[i*4+j] = cudaptr[bx*N+i*4+j];
                    mem[i*4+j] = cuda_weight[i*4+j];
                }
            }
        }
    }

    mem[2*N] = 0;

    __syncthreads();

    if(N<32){
        if(tid<N){
            sum += pow(mem[tid],2);
        }
    }else{
        for(int i = tid;i<f4num;i+=32){
            if(i*4+4<=N){
                float4* tmp = (float4*)(mem+i*4);
                sum += pow(tmp->x,2)+pow(tmp->y,2)+pow(tmp->z,2)+pow(tmp->w,2);
            }else{
                for(int j=0;j<f4remain;j++){
                    sum+= pow(mem[i*4+j],2);
                }
            }
        }
    }
    __syncthreads();

    for(int i=16;i>0;i>>=1){
        float y = __shfl_down_sync(0xFFFFFFFF, sum, i);
        sum +=y;
    }
    if(tid==0){
        mem[2*N] = 1/sqrt((1.f/N)*sum+t);
    }
    __syncthreads();
    if(N<32){
        if(tid<N){
            cudares[bx*N+tid] = mem[tid]*mem[2*N]*mem[N+tid];
        }
    }else{
        for(int i = tid;i<f4num;i+=32){
            if(i*4+4<=N){
                float4 tmp = *(float4*)(mem+i*4);
                float4 weight1 = *(float4*)(mem+N+i*4);
                tmp.x = tmp.x*mem[2*N]*weight1.x;
                tmp.y = tmp.y*mem[2*N]*weight1.y;
                tmp.z = tmp.z*mem[2*N]*weight1.z;
                tmp.w = tmp.w*mem[2*N]*weight1.w;
                
                *(float4*)(cudares+bx*N+i*4) = tmp;


            }else{
                for(int j=0;j<f4remain;j++){
                    cudares[bx*N+i*4+j] = mem[i*4+j]*mem[2*N]*mem[N+i*4+j];
                }
            }
        }
    }

}

void cuda_rms_interface(float* result,float* cudaptr,float* cudares,float* cuda_weight,int M,int N,float t){
    // int blocksize = ((N+3)/4)/
    int blocksize = 32;
    int gridsize = M;
    // int threadpernum = ((N+3)/4)/blocksize;


    cuda_rms<<<gridsize,blocksize,(2*N+1)*sizeof(float)>>>(cudaptr,cudares,cuda_weight,M,N,t);
    cudaMemcpy(result,cudares,M*N*sizeof(float),cudaMemcpyDeviceToHost);
}

void cpu_rms(float* input,float* output,float* weight,int M,int N,float t){
    
    for(int i=0;i<M;i++){
        float sum=0.f;
        for(int j=0;j<N;j++){
            sum+= pow(input[i*N+j],2);
        }

        sum = 1/sqrt((1.0f/N)*sum+t);

        // printf("sum:%f\n",sum);
        for(int j=0;j<N;j++){
            output[i*N+j] = input[i*N+j]*sum*weight[j];
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

void test(int j){
    const float t = 0;
    float* input = (float*)malloc(M*N*sizeof(float));
    for(int i =0;i<M*N;i++){
        input[i] = 1 + i*0.2;
    }

    float* cpu_weight = (float*)malloc(N*sizeof(float));
    float* cuda_weight;
    for(int i =0;i<M;i++){
        cpu_weight[i] = 1;
    }
    float* result = (float*)malloc(M*N*sizeof(float));
    // cudaMalloc(&cudaptr,M*N*sizeof(float));
    // cudaMalloc(&output,M*N*sizeof(float));
    // cudaMemcpy(cudaptr,flash,M*N*sizeof(float));
    float* cudaptr;
    float* cudares;
    cudaMalloc(&cudaptr,M*N*sizeof(float));
    cudaMalloc(&cudares,M*N*sizeof(float));
    cudaMalloc(&cuda_weight,N*sizeof(float));
    cudaMemcpy(cudaptr,input,M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_weight,cpu_weight,N*sizeof(float),cudaMemcpyHostToDevice);



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);
    
    if(j==0){
        cpu_rms(input,answer,cpu_weight,M,N,t);
    }else if(j==1){
        cuda_rms_interface(result,cudaptr,cudares,cuda_weight,M,N,t);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);
    if(j==0){
        printf("cpu_rms elapsed_time:%.4f\n",elapsed_time);
    }else if(j==1){
        printf("cuda_rms elapsed_time:%.4f\n",elapsed_time);
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