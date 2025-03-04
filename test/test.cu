#include"stdio.h"

__global__ void test(){

    printf("hello world from gpu\nthreadid:%d,blockid:%d\n",threadIdx.x,blockIdx.x);

}


int main(){
    test<<<2,12>>>();
    cudaDeviceSynchronize();
    return 0;
}