#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>


#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__constant__ const int TM = 8;
__constant__ const int TN = 8;
__constant__ const int BM = 128;
__constant__ const int BN = 128;
__constant__ const int BK = 8;

__global__ void m_gemm(const int M, const int K, const int N, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)
{


    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    __shared__ float s_a[BK+1][BM+1];
    __shared__ float s_b[BK+1][BN+1]; 
    
    float load_a[4];
    float load_b[4];
    float rc[TM][TN]= {0.0};

    const int tid = ty*blockDim.x+tx;

    int sa_row_mem = tid>>1;
    int sa_col_mem = (tid & 1) << 2;
    int sb_row_mem = tid>>5;
    int sb_col_mem = (tid & 31) << 2;


    int ga_row_mem = by*BM+sa_row_mem;
    int gb_col_mem = bx*BN+sb_col_mem;

    for(int bk = 0;bk < (K+BK-1)/BK;bk++){
        int ga_col_mem = bk*BK + sa_col_mem;
        int ga_addr = OFFSET(ga_row_mem,ga_col_mem,K);

        FLOAT4(load_a[0]) = FLOAT4(A[ga_addr]);
        s_a[sa_col_mem][sa_row_mem] = load_a[0];
        s_a[sa_col_mem+1][sa_row_mem] = load_a[1];
        s_a[sa_col_mem+2][sa_row_mem] = load_a[2];
        s_a[sa_col_mem+3][sa_row_mem] = load_a[3];
        

        
        int gb_row_mem = bk*BK + sb_row_mem;
        int gb_addr = OFFSET(gb_row_mem,gb_col_mem,N);

        
        FLOAT4(load_b[0]) = FLOAT4(B[gb_addr]);
        s_b[sb_row_mem][sb_col_mem]= load_b[0];
        s_b[sb_row_mem][sb_col_mem+1]= load_b[1];
        s_b[sb_row_mem][sb_col_mem+2]= load_b[2];
        s_b[sb_row_mem][sb_col_mem+3]= load_b[3];
        // FLOAT4(s_b[sb_row_mem][sb_col_mem]) = FLOAT4(B[gb_addr]);


        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k++){
            #pragma unroll
            for(int m = 0;m<TM;m++){
                #pragma unroll
                for(int n=0;n<TN;n++){
                    int m_tmp = ty*TM+m;
                    int n_tmp = tx*TN+n;

                    rc[m][n] += s_a[k][m_tmp]*s_b[k][n_tmp];
                    
                }
            }
        }
        __syncthreads();
    }


    #pragma unroll
    for(int i = 0;i<TM;i++){
        
        int gc_row_mem = by*BM+ty*TM+i;
        #pragma unroll
        for(int j = 0;j<TN;j +=4){
            
            int gc_col_mem = bx*BN+tx*TN+j;
            int gc_addr = OFFSET(gc_row_mem,gc_col_mem,N);
            
            FLOAT4(C[gc_addr])=FLOAT4(rc[i][j]);
        }
    }

}


__global__ void m_gemmv3(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K){

    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty*blockDim.x + tx;

    __shared__ float sa_mem[2][BK][BM];
    __shared__ float sb_mem[2][BK][BN];

    float load_a[4];
    float load_b[4];
    // float load_b[4];
    float rc[TM][TN];

    float compute_a[TM];
    float compute_b[TN];

    int sa_row_mem = tid>>1;
    int sa_col_mem = (tid&1)<<2;
    int sb_row_mem = tid>>5;
    int sb_col_mem = (tid&31)<<2;

    int sa_row_gmem = by*BM + sa_row_mem;
    int sb_col_gmem = bx*BN + sb_col_mem;

    {
        int sa_addr = OFFSET(sa_row_gmem,sa_col_mem,K);
        FLOAT4(load_a[0])=FLOAT4(a[sa_addr]);
        int sb_addr = OFFSET(sb_row_mem,sb_col_mem,N);
        FLOAT4(load_b[0]) = FLOAT4(b[sb_addr]);
        sa_mem[0][sa_col_mem][sa_row_mem] = load_a[0];
        sa_mem[0][sa_col_mem+1][sa_row_mem] = load_a[1];
        sa_mem[0][sa_col_mem+2][sa_row_mem] = load_a[2];
        sa_mem[0][sa_col_mem+3][sa_row_mem] = load_a[3];


        FLOAT4(sb_mem[0][sb_row_mem][sb_col_mem]) = FLOAT4(load_b[0]);
    }
    __syncthreads();

    int buffernow = 1;

    for(int bk = 1;bk<(K+BK-1)/BK;bk++){
        buffernow = buffernow ^ 1;
        int buffernext = buffernow^1;

        int sa_col_gmem = bk*BK + sa_col_mem;
        int sa_addr = OFFSET(sa_row_gmem,sa_col_gmem,K);
        FLOAT4(load_a[0])=FLOAT4(a[sa_addr]);
        int sb_row_gmem = bk*BK + sb_row_mem;
        int sb_addr = OFFSET(sb_row_gmem,sb_col_gmem,N);
        FLOAT4(load_b[0]) = FLOAT4(b[sb_addr]);

        #pragma unroll
        for(int k=0;k<BK;k++){
            FLOAT4(compute_a[0])=FLOAT4(sa_mem[buffernow][k][ty * TM / 2]);
            FLOAT4(compute_a[4])=FLOAT4(sa_mem[buffernow][k][ty * TM / 2+BM/2]);

            FLOAT4(compute_b[0])=FLOAT4(sb_mem[buffernow][k][tx * TN / 2]);
            FLOAT4(compute_b[4])=FLOAT4(sb_mem[buffernow][k][tx * TN / 2+BN/2]);

            #pragma unroll
            for(int m=0;m<TM;m++){
                #pragma unroll
                for(int n=0;n<TN;n++){
                    rc[m][n] += compute_a[m]*compute_b[n];
                }
            }

        }



        sa_mem[buffernext][sa_col_mem][sa_row_mem] = load_a[0];
        sa_mem[buffernext][sa_col_mem+1][sa_row_mem] = load_a[1];
        sa_mem[buffernext][sa_col_mem+2][sa_row_mem] = load_a[2];
        sa_mem[buffernext][sa_col_mem+3][sa_row_mem] = load_a[3];

        FLOAT4(sb_mem[buffernext][sb_row_mem][sb_col_mem]) = FLOAT4(load_b[0]);

        __syncthreads();
    }

    buffernow = buffernow ^ 1;
    #pragma unroll
    for(int k=0;k<BK;k++){
        FLOAT4(compute_a[0])=FLOAT4(sa_mem[buffernow][k][ty * TM / 2]);
        FLOAT4(compute_a[4])=FLOAT4(sa_mem[buffernow][k][ty * TM / 2+BM/2]);

        FLOAT4(compute_b[0])=FLOAT4(sb_mem[buffernow][k][tx * TN / 2]);
        FLOAT4(compute_b[4])=FLOAT4(sb_mem[buffernow][k][tx * TN / 2+BN/2]);

        #pragma unroll
        for(int m=0;m<TM;m++){
            #pragma unroll
            for(int n=0;n<TN;n++){
                rc[m][n] += compute_a[m]*compute_b[n];
            }
        }

    }
    __syncthreads();

    #pragma unroll
    for(int i =0;i<TM/2;i++){
        int c_col_gmem = bx*BN + tx*TN/2;
        int c_row_gmem = by*BM + ty*TM/2 + i;
        int c_addr = OFFSET(c_row_gmem,c_col_gmem,N);

        FLOAT4(c[c_addr])=FLOAT4(rc[i][0]);
        FLOAT4(c[c_addr+BN/2])=FLOAT4(rc[i][4]);

    }

    #pragma unroll
    for(int i =0;i<TM/2;i++){
        int c_col_gmem = bx*BN + tx*TN/2;
        int c_row_gmem = by*BM +BM/2 +ty*TM/2 + i;
        int c_addr = OFFSET(c_row_gmem,c_col_gmem,N);

        FLOAT4(c[c_addr])=FLOAT4(rc[i][0]);
        FLOAT4(c[c_addr+BN/2])=FLOAT4(rc[i][4]);

    }

}


void print_matrix(float* matrix,int length,int width){
    for(int i = 0;i<width;i++){
        for(int j=0;j<length;j++){
            int addr = i*length + j;
            printf("%.1f, ",matrix[addr]);
        }
        printf("\n");
    }
}

__global__ void easy_gemm(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K){
    int row = blockIdx.x;
    int col = threadIdx.x;

    float sum = 0.f;
    for(int i=0;i<K;i++){
        sum+=a[row*K+i] * b[i*N+col];
    }
    c[row*N+col]=sum;
}

void easy_gemm_interface(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K){
    const dim3 blocksize(N);
    const dim3 gridsize(M);
    
    easy_gemm<<<gridsize,blocksize>>>(a, b, c,M, N,K);
    

}


void checkcompute(float* a,float* b,int size){
    for(int i=0;i<size;i++){
        if(a[i]!=b[i]){
            printf("error!!   ||answer[%d]:%f,result[%d]:%f\n",i,a[i],i,b[i]);
            return;
        }
    }
    printf("success\n");
}
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

static float* answer=(float*)malloc(M*N*sizeof(float));
// #MxK * KxN
void test(int i){
    



    float* h_A = (float*)malloc(M*K*sizeof(float));
    float* h_B = (float*)malloc(N*K*sizeof(float));
    float* h_C = (float*)malloc(M*N*sizeof(float));

    const dim3 blocksize(BN/TN,BM/TM);
    const dim3 gridsize(N/BN,M/BM);

    for(int i = 0;i<M*K;i++){
        h_A[i] = 2;
    }

    for(int i = 0;i<N*K;i++){
        h_B[i] = 3;
    }

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A , M*K*sizeof(float));
    cudaMalloc(&d_B , N*K*sizeof(float));
    cudaMalloc(&d_C , M*N*sizeof(float));

    cudaMemcpy( d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_B, h_B, N*K*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);


    if(i==1){
        m_gemm<<<gridsize,blocksize>>>(M,K,N,d_A,d_B,d_C);
    }else if(i==2){
        m_gemmv3<<<gridsize,blocksize>>>(d_A, d_B, d_C,M, N,K);
    }else if(i==3){
        easy_gemm_interface(d_A, d_B, d_C,M, N,K);
    }



    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);

    if(i==1){
        printf("m_gemm elasped_time:%.4f\n",elapsed_time);
        cudaMemcpy(answer, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);
    }else if(i==2){
        printf("m_gemmv3 elasped_time:%.4f\n",elapsed_time);
    }else if(i==3){
        printf("easy_gemm elasped_time:%.4f\n",elapsed_time);
        float* result=(float*)malloc(M*N*sizeof(float));
        cudaMemcpy(result,d_C,M*K*sizeof(float), cudaMemcpyDeviceToHost);
        checkcompute(answer,result,M*K);
    }


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy( h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


// __global__ bank_test(    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
//     const int M, const int N, const int K){
    
// }


int main()
{
    for(int j=0;j<7;j++){
        printf("round %d\n",j+1);
        for(int i=1;i<=5;i++){
            test(i);
        }
        printf("\n");
    }

    return 0;
}