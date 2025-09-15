#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
using namespace cooperative_groups;

const int head_size = 512;
const int head_num = 32;
const int pos = 10;
const int seq_len = 15;
const int kv_mul = 8;
int kv_dim = kv_mul*head_size;
int N=head_num*head_size;
float* answer = (float*)malloc(N*sizeof(float));

__global__ void flash_attention(int32_t pos, int32_t seq_len, float* query,float* output, float* key_cache,
            float* value_cache, int32_t kv_dim, int32_t kv_mul,int32_t head_num, int32_t head_size,int32_t layer_offset){
            extern __shared__ float mem[];
            for(int i=0;i<head_size;i++){
                mem[i]=0.f;
            }

            int head = blockIdx.x;
            int f4num = head_size/4;
            int f4remain = head_size%4;
            float scale = 1.f / sqrtf(float(head_size));  
            int head_offset = head_size*(head/kv_mul);
            float* query_head = query+head*head_size;
            float lastsum=0.f;
            float lastmax=-INFINITY;
            // float lastoutput=0.f;
            for(int i=0;i<pos;i++){
                float* key_head = key_cache + i*kv_dim + head_offset;
                float tmp3=0.f;
                for(int j=0;j<f4num;j++){
                    float4 tmp1 = *(float4*)(query_head+j*4);
                    float4 tmp2 = *(float4*)(key_head+4*j);
                    tmp3 += (tmp1.x)*(tmp2.x);
                    tmp3 += (tmp1.y)*(tmp2.y);
                    tmp3 += (tmp1.z)*(tmp2.z);
                    tmp3 += (tmp1.w)*(tmp2.w);
                }
                for(int j=0;j<f4remain;j++){
                    float tmp1 = *(query_head+j+f4num*4);
                    float tmp2 = *(key_head+j+f4num*4);
                    tmp3+=tmp1*tmp2;
                }
                tmp3 =tmp3*scale;
                float max = lastmax>=tmp3? lastmax:tmp3;
                tmp3 = expf(tmp3-max);
                float sum = lastsum*expf(lastmax-max)+tmp3;
                
                float* value_head = value_cache+i*kv_dim+head_offset;
                for(int j=0;j<f4num;j++){
                    float4 tmp1 = *(float4*)(value_head+j*4);
                    float4* tmp2 = (float4*)(mem+j*4);
                    tmp2->x = (1.f/sum)*(tmp2->x*lastsum*expf(lastmax-max)+tmp1.x*tmp3);
                    tmp2->w = (1.f/sum)*(tmp2->w*lastsum*expf(lastmax-max)+tmp1.w*tmp3);
                    tmp2->y = (1.f/sum)*(tmp2->y*lastsum*expf(lastmax-max)+tmp1.y*tmp3);
                    tmp2->z = (1.f/sum)*(tmp2->z*lastsum*expf(lastmax-max)+tmp1.z*tmp3);
                }
                for(int j=0;j<f4remain;j++){
                    float tmp1 = *(value_head+4*f4num+j);
                    float* tmp2 = (mem+4*f4num+j);
                    *tmp2 = (1.f/sum)*((*tmp2)*lastsum*expf(lastmax-max)+tmp1*tmp3);
                }
                lastmax = max;
                lastsum = sum;
            }
            float* output_head = output+head*head_size;
            for(int j=0;j<f4num;j++){
                *(float4*)(output_head+j*4) = *(float4*)(mem+j*4);
            }
            for(int j=0;j<f4remain;j++){
                *(output_head+4*f4num+j) = *(mem+4*f4num+j);
            }            
}


// lastsum,lastmax,
void flash_attention_interface(int32_t pos, int32_t seq_len, float* query,float* output, float* key_cache,
            float* value_cache, int32_t kv_dim, int32_t kv_mul,int32_t head_num, int32_t head_size,int32_t layer_offset){
            int blocksize = 1;
            int gridsize = head_num;

            flash_attention<<<gridsize,blocksize,head_size*sizeof(float)>>>(pos, seq_len, query,output, key_cache,
                                                    value_cache, kv_dim, kv_mul,head_num, head_size,layer_offset);
}

void softmax_cpu(float* input,int size){

    float max = 0.f;
    for(int i=0;i<size;i++){
        max = max>= *(input+i) ? max : *(input+i);
    }

    float sum=0.f;
    for(int i=0;i<size;i++){
        sum+=expf(*(input+i)-max);
    }

    for(int i=0;i<size;i++){
        *(input+i)=expf(*(input+i)-max)/sum;
    }
}

void cpu_mha(int32_t pos, int32_t seq_len, float* query,float* output, float* key_cache,
            float* value_cache, int32_t kv_dim, int32_t kv_mul,int32_t head_num, int32_t head_size,int32_t layer_offset){
        float* score_ptr = (float*)malloc(head_num*pos*sizeof(float)); 
        float scale = 1.f / sqrtf(float(head_size));       
        // float scale = 1.f; 
        for(int head=0;head<head_num;head++){
            int head_offset = (head/kv_mul)*head_size;
            float* query_head = query+head*head_size;
            
            for(int idx=0;idx<pos;idx++){
                float* key_head = key_cache+idx*kv_dim+head_offset;
                float sum=0.f;
                for(int j=0;j<head_size;j++){
                    sum += *(query_head+j)*(*(key_head+j));
                }
                *(score_ptr+head*pos+idx) = sum*scale;
            }

            softmax_cpu(score_ptr+head*pos,pos);


            for(int idx=0;idx<pos;idx++){
                float* Value_head = value_cache+idx*kv_dim+head_offset;

                
                for(int j=0;j<head_size;j++){
                    *(output+head*head_size+j) +=*(score_ptr+head*pos+idx)*(*(Value_head+j));
                }

            }

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
    float* Query = (float*)malloc(N*sizeof(float));
    for(int i =0;i<N;i++){
        Query[i] = i;
    }

    float* Key = (float*)malloc(seq_len*kv_dim*sizeof(float));
    float* Value = (float*)malloc(seq_len*kv_dim*sizeof(float));
    for(int i =0;i<seq_len*kv_dim;i++){
        Key[i] = i;
        Value[i] = i;
    }
    int layer_offset = 0;

    float* cuda_key;
    float* cuda_query;
    float* cuda_value;
    float* cuda_output;
    float* output = (float*)malloc(N*sizeof(float));
    cudaMalloc(&cuda_key,seq_len*kv_dim*sizeof(float));
    cudaMalloc(&cuda_value,seq_len*kv_dim*sizeof(float));
    cudaMalloc(&cuda_query,N*sizeof(float));
    cudaMalloc(&cuda_output,N*sizeof(float));

    cudaMemcpy(cuda_key,Key,seq_len*kv_dim*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_value,Value,seq_len*kv_dim*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_query,Query,N*sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);
    
    if(j==0){
        cpu_mha(pos, seq_len, Query,answer, Key,
            Value, kv_dim, kv_mul,head_num, head_size,layer_offset);
    }else if(j==1){
        flash_attention_interface(pos, seq_len, cuda_query,cuda_output, cuda_key,
            cuda_value, kv_dim, kv_mul,head_num, head_size,layer_offset);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);

    
    if(j==0){
        printf("cpu_mha elapsed_time:%.4f\n",elapsed_time);
    }else if(j==1){
        printf("cuda_mha elapsed_time:%.4f\n",elapsed_time);
        cudaMemcpy(output,cuda_output,N*sizeof(float),cudaMemcpyDeviceToHost);
        checkcompute(answer,output,N);
        // printf("computecheck over\n");
    }
    free(output);
    free(Query);
    free(Key);
    free(Value);
    cudaFree(cuda_query);
    cudaFree(cuda_key); 
    cudaFree(cuda_value); 
    cudaFree(cuda_output); 
}
int main(){
        // printf("M:%d,N:%d\n",M,N);
    // memset(answer, 0, head_num * head_size * sizeof(float));
    
    for(int i=0;i<3;i++){
        printf("round:%d\n",i);
        memset(answer, 0, head_num * head_size * sizeof(float));
        for(int j=0;j<2;j++){
            test(j);
        }
    }
    // print_matrix(answer,N,1);
    return 0;
}